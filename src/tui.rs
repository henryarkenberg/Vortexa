//! A Ratatui terminal UI for Vortexa: menu, chat, live training gauge,
//! a settings panel, device selection, and an about screen.
//!
//! Training runs on a background thread and reports progress through a
//! shared `TrainProgress`, so the UI can draw a real-time gauge. The plain
//! line-based menu in `ui.rs` remains available as a fallback.

use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use rand::{rngs::StdRng, SeedableRng};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Gauge, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::config::Config;
use crate::datasets::{self, DATASETS, DownloadProgress};
use crate::device;
use crate::generate::{self, Generator};
use crate::settings::Settings;
use crate::train::{self, TrainArgs, TrainProgress};

const MENU_ITEMS: [&str; 9] = [
    "Train",
    "Continue",
    "Chat / Ask",
    "Datasets",
    "Evaluate",
    "Settings",
    "Device",
    "About",
    "Exit",
];

#[derive(Clone, Copy, PartialEq)]
enum Screen {
    Menu,
    TrainForm,
    TrainRun,
    Chat,
    Datasets,
    Eval,
    Settings,
    About,
}

// One editable line in the settings panel.
#[derive(Clone)]
struct Entry {
    label: String,
    value: String,
}

pub struct App {
    screen: Screen,
    menu_index: usize,
    quit: bool,
    version: String,
    settings: Settings,

    // Settings panel state
    settings_entries: Vec<Entry>,
    settings_selected: usize,
    settings_editing: bool,
    edit_buffer: String,

    // Train form + background run
    train_data: String,
    train_steps: String,
    train_field: usize,
    train_resume: Option<PathBuf>,
    data_files: Vec<String>,
    train_progress: Arc<Mutex<TrainProgress>>,
    train_cancel: Arc<AtomicBool>,
    train_handle: Option<thread::JoinHandle<()>>,

    // Chat
    chat_generator: Option<Generator>,
    chat_history: Vec<String>,
    chat_input: String,
    chat_rng: StdRng,
    chat_scroll: usize,
    chat_scroll_pending: bool,

    // Datasets
    dataset_index: usize,
    download_progress: Arc<Mutex<DownloadProgress>>,
    download_cancel: Arc<AtomicBool>,
    download_handle: Option<thread::JoinHandle<()>>,
}

impl App {
    fn new() -> Self {
        let settings = Settings::load(Path::new("."));
        let settings_entries = entries_from_settings(&settings);
        Self {
            screen: Screen::Menu,
            menu_index: 0,
            quit: false,
            version: env!("CARGO_PKG_VERSION").to_string(),
            settings,
            settings_entries,
            settings_selected: 0,
            settings_editing: false,
            edit_buffer: String::new(),
            train_data: "data/input.txt".to_string(),
            train_steps: "20000".to_string(),
            train_field: 0,
            train_resume: None,
            data_files: datasets::data_files(),
            train_progress: Arc::new(Mutex::new(TrainProgress::default())),
            train_cancel: Arc::new(AtomicBool::new(false)),
            train_handle: None,
            chat_generator: None,
            chat_history: Vec::new(),
            chat_input: String::new(),
            chat_rng: StdRng::seed_from_u64(42),
            chat_scroll: 0,
            chat_scroll_pending: false,
            dataset_index: 0,
            download_progress: Arc::new(Mutex::new(DownloadProgress::default())),
            download_cancel: Arc::new(AtomicBool::new(false)),
            download_handle: None,
        }
    }

    fn device_name(&self) -> String {
        device::pick_device(&self.settings.device)
            .map(|d| device::describe(&d))
            .unwrap_or_else(|_| self.settings.device.clone())
    }

    fn build_config(&self) -> Config {
        let s = &self.settings;
        Config {
            vocab_size: 256,
            d_model: s.d_model,
            num_layers: s.layers,
            num_heads: s.heads,
            head_dim: s.head_dim,
            ffn_dim: s.ffn,
            max_seq_len: s.seq_len,
            decay_min: s.decay_min,
            decay_max: s.decay_max,
            chunk_len: s.chunk_len,
            tokenizer: s.tokenizer.clone(),
            num_merges: s.num_merges,
        }
    }

    fn start_training(&mut self) {
        // Re-list `data/` so the picked file matches what is actually on disk.
        self.refresh_data_files();
        let steps = self.train_steps.trim().parse::<usize>().unwrap_or(self.settings.steps);
        let config = match &self.train_resume {
            Some(_) => config_from_checkpoint(Path::new("checkpoints")),
            None => self.build_config(),
        };
        let data = if self.train_data.trim().is_empty() {
            self.settings.data_file.clone()
        } else {
            self.train_data.trim().to_string()
        };
        let s = &self.settings;
        let args = TrainArgs {
            data: PathBuf::from(data),
            out_dir: PathBuf::from("checkpoints"),
            steps,
            batch_size: s.batch_size,
            seq_len: s.seq_len,
            lr: s.lr,
            log_every: 50,
            val_every: 500,
            val_batches: 4,
            save_every: 1000,
            val_frac: 0.05,
            seed: 42,
            resume: self.train_resume.clone(),
            warmup_steps: 200,
            grad_clip: 1.0,
            device: s.device.clone(),
            config,
        };

        let progress = Arc::new(Mutex::new(TrainProgress::default()));
        let cancel = Arc::new(AtomicBool::new(false));
        let p = progress.clone();
        let c = cancel.clone();
        self.train_handle = Some(thread::spawn(move || {
            // Surface setup errors (missing data, bad config, ...) and panics
            // so the UI shows why instead of a stuck 0/0 gauge.
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                train::run_with_progress(args, p.clone(), c)
            }));
            match res {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    let mut g = p.lock().unwrap();
                    g.error = true;
                    g.done = true;
                    g.error_msg = format!("{e:#}");
                }
                Err(_) => {
                    let mut g = p.lock().unwrap();
                    g.error = true;
                    g.done = true;
                    g.error_msg = "training panicked unexpectedly".to_string();
                }
            }
        }));
        self.train_progress = progress;
        self.train_cancel = cancel;
        self.screen = Screen::TrainRun;
    }

    fn start_download(&mut self) {
        let dataset = DATASETS[self.dataset_index];
        let progress = Arc::new(Mutex::new(DownloadProgress::default()));
        let cancel = Arc::new(AtomicBool::new(false));
        let p = progress.clone();
        let c = cancel.clone();
        self.download_handle = Some(thread::spawn(move || {
            let _ = datasets::download(&dataset, Path::new("data"), p, c);
        }));
        self.download_progress = progress;
        self.download_cancel = cancel;
        self.data_files = datasets::data_files();
    }

    fn ensure_chat_generator(&mut self) {
        if self.chat_generator.is_some() {
            return;
        }
        let dev = device::pick_device(&self.settings.device).unwrap_or(candle_core::Device::Cpu);
        match Generator::load(Path::new("checkpoints"), &dev) {
            Ok(g) => self.chat_generator = Some(g),
            Err(e) => self
                .chat_history
                .push(format!(
                    "Model not loaded: {e:#}\nTrain one first (menu: Train or Continue)."
                )),
        }
    }

    fn on_enter(&mut self) {
        match self.screen {
            Screen::Menu => match self.menu_index {
                0 => {
                    self.screen = Screen::TrainForm;
                    self.train_resume = None;
                    self.train_steps = self.settings.steps.to_string();
                    self.refresh_data_files();
                }
                1 => {
                    self.screen = Screen::TrainForm;
                    self.train_resume = Some(PathBuf::from("checkpoints"));
                    self.train_steps = "5000".to_string();
                    self.refresh_data_files();
                }
                2 => {
                    self.ensure_chat_generator();
                    self.screen = Screen::Chat;
                }
                3 => {
                    self.dataset_index = 0;
                    self.screen = Screen::Datasets;
                }
                4 => self.screen = Screen::Eval,
                5 => {
                    self.settings_entries = entries_from_settings(&self.settings);
                    self.settings_selected = 0;
                    self.settings_editing = false;
                    self.screen = Screen::Settings;
                }
                6 => self.cycle_device(),
                7 => self.screen = Screen::About,
                _ => self.request_quit(),
            },
            Screen::TrainForm => self.start_training(),
            Screen::Chat => self.submit_chat(),
            Screen::Datasets => self.start_download(),
            Screen::Settings if self.settings_editing => self.commit_edit(),
            _ => {}
        }
    }

    fn cycle_device(&mut self) {
        let order = ["auto", "cpu", "cuda", "metal"];
        let idx = order
            .iter()
            .position(|d| *d == self.settings.device)
            .unwrap_or(0);
        self.settings.device = order[(idx + 1) % order.len()].to_string();
        let _ = self.settings.save(Path::new("."));
    }

    fn submit_chat(&mut self) {
        let prompt = self.chat_input.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        self.chat_input.clear();
        // Always scroll to whatever was just produced (an answer or an error),
        // even when no model is loaded yet.
        self.chat_scroll_pending = true;
        self.ensure_chat_generator();
        let generator = match self.chat_generator.as_mut() {
            Some(g) => g,
            None => return,
        };
        let s = &self.settings;
        let full = generate::apply_template(&s.chat_template, &prompt);
        match generator.complete(&full, s.tokens, s.temperature, s.top_k, &mut self.chat_rng) {
            Ok(text) => self.chat_history.push(format!("You: {prompt}\n\n{text}")),
            Err(e) => self.chat_history.push(format!("[error] {e:#}")),
        }
    }

    fn commit_edit(&mut self) {
        let i = self.settings_selected;
        if let Some(entry) = self.settings_entries.get_mut(i) {
            entry.value = self.edit_buffer.clone();
        }
        self.settings_editing = false;
        // Re-apply (invalid numeric edits fall back to their previous value).
        let new = apply_settings(&self.settings_entries, &self.settings);
        self.settings = new;
        // Manually editing an architecture field means the template no longer
        // describes the model, so switch to "custom".
        if (10..=14).contains(&i) {
            self.settings.model_template = "custom".to_string();
        }
        self.settings_entries = entries_from_settings(&self.settings);
        let _ = self.settings.save(Path::new("."));
    }

    fn cycle_data(&mut self, dir: isize) {
        if self.data_files.is_empty() {
            return;
        }
        // The displayed value is "data/<file>"; find which entry matches.
        let current = self
            .train_data
            .strip_prefix("data/")
            .map(|s| s.to_string())
            .unwrap_or_default();
        let idx = self.data_files.iter().position(|f| *f == current).unwrap_or(0);
        let n = self.data_files.len() as isize;
        let next = (idx as isize + dir).rem_euclid(n) as usize;
        self.train_data = format!("data/{}", self.data_files[next]);
    }

    fn refresh_data_files(&mut self) {
        self.data_files = datasets::data_files();
        if let Some(first) = self.data_files.first() {
            if !self.data_files.contains(&self.train_data.strip_prefix("data/").unwrap_or("").to_string()) {
                self.train_data = format!("data/{first}");
            }
        }
    }

    fn on_key(&mut self, code: KeyCode, ctrl: bool) {
        if ctrl && code == KeyCode::Char('c') {
            self.request_quit();
            return;
        }
        match self.screen {
            Screen::Menu => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.menu_index = self.menu_index.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.menu_index = (self.menu_index + 1).min(MENU_ITEMS.len() - 1)
                }
                KeyCode::Enter => self.on_enter(),
                KeyCode::Char('q') | KeyCode::Esc => self.request_quit(),
                _ => {}
            },
            Screen::TrainForm => match code {
                KeyCode::Tab => self.train_field = (self.train_field + 1) % 2,
                KeyCode::Up | KeyCode::Char('k') if self.train_field == 0 => self.cycle_data(-1),
                KeyCode::Down | KeyCode::Char('j') if self.train_field == 0 => self.cycle_data(1),
                KeyCode::Backspace => self.backspace_train(),
                KeyCode::Char(c) => self.typed_train(c),
                KeyCode::Enter => self.start_training(),
                KeyCode::Esc => self.screen = Screen::Menu,
                _ => {}
            },
            Screen::TrainRun => match code {
                KeyCode::Esc | KeyCode::Char('q') => self.cancel_training(),
                _ => {}
            },
            Screen::Chat => match code {
                KeyCode::Backspace => {
                    self.chat_input.pop();
                }
                KeyCode::Enter => self.submit_chat(),
                KeyCode::Esc => self.screen = Screen::Menu,
                KeyCode::PageUp | KeyCode::Char('k') => {
                    self.chat_scroll = self.chat_scroll.saturating_add(3)
                }
                KeyCode::PageDown | KeyCode::Char('j') => {
                    self.chat_scroll = self.chat_scroll.saturating_sub(3);
                    self.chat_scroll_pending = false;
                }
                KeyCode::Char(c) => self.chat_input.push(c),
                _ => {}
            },
            Screen::Settings => self.on_settings_key(code),
            Screen::Datasets => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.dataset_index = self.dataset_index.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.dataset_index = (self.dataset_index + 1).min(DATASETS.len() - 1)
                }
                KeyCode::Enter => self.start_download(),
                KeyCode::Esc | KeyCode::Char('q') => self.cancel_download_and_back(),
                _ => {}
            },
            Screen::Eval | Screen::About => {
                if code == KeyCode::Esc || code == KeyCode::Char('q') {
                    self.screen = Screen::Menu;
                }
            }
        }
    }

    fn cancel_download_and_back(&mut self) {
        self.download_cancel.store(true, Ordering::Relaxed);
        if let Some(h) = self.download_handle.take() {
            let _ = h.join();
        }
        self.screen = Screen::Menu;
    }

    fn on_settings_key(&mut self, code: KeyCode) {
        // The template row (index 0) is not a free-text edit: Enter/arrows
        // cycle through the size presets and apply them.
        let is_template_row = self.settings_selected == 0;
        if self.settings_editing {
            match code {
                KeyCode::Backspace => {
                    self.edit_buffer.pop();
                }
                KeyCode::Char(c) => self.edit_buffer.push(c),
                KeyCode::Enter => self.commit_edit(),
                KeyCode::Esc => self.settings_editing = false,
                _ => {}
            }
        } else if is_template_row {
            match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.settings_selected = self.settings_selected.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.settings_selected =
                        (self.settings_selected + 1).min(self.settings_entries.len() - 1)
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Left => self.cycle_template(),
                KeyCode::Esc => self.screen = Screen::Menu,
                _ => {}
            }
        } else {
            match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.settings_selected = self.settings_selected.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.settings_selected =
                        (self.settings_selected + 1).min(self.settings_entries.len() - 1)
                }
                KeyCode::Enter => {
                    self.edit_buffer = self.settings_entries[self.settings_selected].value.clone();
                    self.settings_editing = true;
                }
                KeyCode::Esc => {
                    self.screen = Screen::Menu;
                }
                _ => {}
            }
        }
    }

    /// Cycle to the next model-size template and apply its architecture.
    fn cycle_template(&mut self) {
        let cur = self.settings.model_template.clone();
        let idx = TEMPLATES.iter().position(|t| t.name == cur).unwrap_or(0);
        let next = TEMPLATES[(idx + 1) % TEMPLATES.len()];
        let s = &mut self.settings;
        s.model_template = next.name.to_string();
        s.d_model = next.d_model;
        s.layers = next.layers;
        s.heads = next.heads;
        s.head_dim = next.head_dim;
        s.ffn = next.ffn;
        s.chunk_len = 64;
        self.settings_entries = entries_from_settings(&self.settings);
        let _ = self.settings.save(Path::new("."));
    }

    fn backspace_train(&mut self) {
        if self.train_field == 0 {
            self.train_data.pop();
        } else {
            self.train_steps.pop();
        }
    }

    fn typed_train(&mut self, c: char) {
        if self.train_field == 0 {
            self.train_data.push(c);
        } else if c.is_ascii_digit() {
            self.train_steps.push(c);
        }
    }

    fn cancel_training(&mut self) {
        self.train_cancel.store(true, Ordering::Relaxed);
        if let Some(h) = self.train_handle.take() {
            let _ = h.join();
        }
        self.screen = Screen::Menu;
    }

    fn request_quit(&mut self) {
        self.train_cancel.store(true, Ordering::Relaxed);
        if let Some(h) = self.train_handle.take() {
            let _ = h.join();
        }
        self.quit = true;
    }

    fn draw(&mut self, f: &mut Frame) {
        match self.screen {
            Screen::Menu => self.draw_menu(f),
            Screen::TrainForm => self.draw_train_form(f),
            Screen::TrainRun => self.draw_train_run(f),
            Screen::Chat => self.draw_chat(f),
            Screen::Datasets => self.draw_datasets(f),
            Screen::Eval => self.draw_eval(f),
            Screen::Settings => self.draw_settings(f),
            Screen::About => self.draw_about(f),
        }
    }

    fn draw_menu(&mut self, f: &mut Frame) {
        let area = f.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Min(8),
                Constraint::Length(3),
            ])
            .split(area);

        f.render_widget(
            Paragraph::new(Text::raw(banner_text())).alignment(Alignment::Center),
            chunks[0],
        );

        let items: Vec<ListItem> = MENU_ITEMS.iter().map(|i| ListItem::new(*i)).collect();
        let list = List::new(items)
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title(" Main menu ")
                    .title_alignment(Alignment::Center),
            )
            .highlight_style(
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("> ");
        let mut state = ListState::default();
        state.select(Some(self.menu_index));
        f.render_stateful_widget(list, chunks[1], &mut state);

        let status = format!(
            "Device: {}   ·   v{}   ·   ↑/↓ or j/k move, Enter select, q quit",
            self.device_name(),
            self.version
        );
        f.render_widget(
            Paragraph::new(status).alignment(Alignment::Center),
            chunks[2],
        );
    }

    fn draw_train_form(&mut self, f: &mut Frame) {
        let area = f.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(4), Constraint::Length(2)])
            .split(area);
        f.render_widget(
            Paragraph::new(Text::raw(
                "Enter training settings. Tab switches field, Enter starts, Esc back.",
            )),
            chunks[0],
        );

        let (data_style, steps_style) = if self.train_field == 0 {
            (Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD), Style::default())
        } else {
            (Style::default(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        };
        let files_hint = if self.data_files.is_empty() {
            "no .txt files in data/ yet (use the Datasets menu)".to_string()
        } else {
            format!("available: {}", self.data_files.join(", "))
        };
        let lines = vec![
            Line::from(vec![
                Span::styled("Data file: ", Style::default()),
                Span::styled(self.train_data.clone(), data_style),
            ]),
            Line::from(vec![
                Span::styled("Steps: ", Style::default()),
                Span::styled(self.train_steps.clone(), steps_style),
            ]),
            Line::from(Span::raw(files_hint)),
        ];
        f.render_widget(
            Paragraph::new(Text::from(lines))
                .block(Block::bordered().title(" Train ").title_alignment(Alignment::Center)),
            chunks[1],
        );
        f.render_widget(
            Paragraph::new(Text::raw("Esc back   ·   Tab field   ·   Enter start")),
            chunks[2],
        );
    }

    fn draw_train_run(&mut self, f: &mut Frame) {
        let area = f.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(6), Constraint::Length(3)])
            .split(area);

        let prog = self.train_progress.lock().unwrap().clone();
        let percent = if prog.total > 0 {
            ((prog.step as f64 / prog.total as f64) * 100.0).min(100.0) as u16
        } else {
            0
        };
        let label = if prog.error {
            "error".to_string()
        } else if prog.total > 0 {
            format!(
                "step {}/{}   loss {:.3}   {:.1}k tok/s",
                prog.step,
                prog.total,
                prog.loss,
                prog.tok_s / 1e3
            )
        } else {
            "preparing data...".to_string()
        };
        f.render_widget(
            Gauge::default()
                .block(
                    Block::bordered()
                        .title(" Training ")
                        .title_alignment(Alignment::Center),
                )
                .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
                .percent(percent)
                .label(label),
            chunks[0],
        );

        let status = if prog.error {
            format!("Error: {}", prog.error_msg)
        } else if prog.done {
            "Status: finished   ·   Esc to return".to_string()
        } else if prog.total == 0 {
            "Preparing dataset / tokenizer...   ·   Esc to stop".to_string()
        } else {
            "Status: running   ·   Esc or q to stop".to_string()
        };
        f.render_widget(
            Paragraph::new(Text::raw(status).alignment(Alignment::Center)),
            chunks[1],
        );
    }

    fn draw_chat(&mut self, f: &mut Frame) {
        let area = f.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(area);

        let lines: Vec<Line> = self
            .chat_history
            .iter()
            .flat_map(|h| h.lines().map(|l| Line::from(l.to_string())))
            .collect();

        // Auto-scroll to the bottom when a new message arrives.
        let total = lines.len();
        let visible = chunks[0].height as usize;
        if self.chat_scroll_pending {
            self.chat_scroll = total.saturating_sub(visible);
            self.chat_scroll_pending = false;
        }

        f.render_widget(
            Paragraph::new(Text::from(lines))
                .block(Block::bordered().title(" Chat ").title_alignment(Alignment::Center))
                .wrap(Wrap { trim: false })
                .scroll((self.chat_scroll as u16, 0)),
            chunks[0],
        );
        f.render_widget(
            Paragraph::new(self.chat_input.clone())
                .block(Block::bordered().title(" Ask ").title_alignment(Alignment::Center))
                .style(Style::default().add_modifier(Modifier::BOLD)),
            chunks[1],
        );
    }

    fn draw_settings(&self, f: &mut Frame) {
        let area = f.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(6), Constraint::Length(1)])
            .split(area);
        let items: Vec<ListItem> = self
            .settings_entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let value = if self.settings_editing && i == self.settings_selected {
                    self.edit_buffer.clone()
                } else {
                    e.value.clone()
                };
                ListItem::new(format!("{}: {}", e.label, value))
            })
            .collect();
        let list = List::new(items)
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title(" Settings ")
                    .title_alignment(Alignment::Center),
            )
            .highlight_style(
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("> ");
        let mut state = ListState::default();
        state.select(Some(self.settings_selected));
        f.render_stateful_widget(list, chunks[0], &mut state);

        // Tiny/small/medium/large are ~0.5M / 2.8M / 7M / 17M parameters.
        let hint = if self.settings_selected == 0 {
            "Enter or ←/→: cycle size template (tiny .5M, small 2.8M, medium 7M, large 17M)   ·   Esc back"
        } else {
            "Enter: edit value   ·   ↑/↓ move   ·   Esc back"
        };
        f.render_widget(Paragraph::new(hint), chunks[1]);
    }

    fn draw_datasets(&mut self, f: &mut Frame) {
        let area = f.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(7), Constraint::Length(3)])
            .split(area);

        let downloading = self.download_handle.is_some();
        let items: Vec<ListItem> = DATASETS
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let marker = if i == self.dataset_index { "> " } else { "  " };
                let present = Path::new("data").join(d.file).exists();
                let tag = if present { "  [downloaded]" } else { "" };
                ListItem::new(format!(
                    "{marker}{:<20} {:<9} {}{}",
                    d.name, d.size_hint, d.desc, tag
                ))
            })
            .collect();
        let list = List::new(items)
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title(" Datasets ")
                    .title_alignment(Alignment::Center),
            )
            .highlight_style(
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("");
        let mut state = ListState::default();
        state.select(Some(self.dataset_index));
        f.render_stateful_widget(list, chunks[0], &mut state);

        let status = if downloading {
            let d = *self.download_progress.lock().unwrap();
            let name = DATASETS[self.dataset_index].name;
            if d.finished {
                "Download complete   ·   Esc to go back".to_string()
            } else {
                format!(
                    "Downloading {name} ... {:.1} / {} MB   ·   Esc to cancel",
                    d.done as f64 / 1e6,
                    if d.total > 0 { d.total as f64 / 1e6 } else { 0.0 }
                )
            }
        } else {
            "Enter: download to data/   ·   ↑/↓ select   ·   Esc back".to_string()
        };
        f.render_widget(Paragraph::new(status).alignment(Alignment::Center), chunks[1]);
    }

    fn draw_eval(&mut self, f: &mut Frame) {
        let text = format!(
            "Deterministic evaluation\n\nRun this from a normal shell for the numbers:\n\n   vortexa eval --checkpoint checkpoints\n\nDevice: {}\n\nEsc to return.",
            self.device_name()
        );
        f.render_widget(
            Paragraph::new(Text::raw(text)).wrap(Wrap { trim: false }),
            f.size(),
        );
    }

    fn draw_about(&mut self, f: &mut Frame) {
        let text = format!(
            "Vortexa {}\n\nA tiny RetNet language model in Rust with Candle.\nRetention, not attention. No KV cache.\nTrain it on any text, chat with it, measure it.\n\nMIT licensed. Esc to return.",
            self.version
        );
        f.render_widget(
            Paragraph::new(Text::raw(text)).wrap(Wrap { trim: false }),
            f.size(),
        );
    }
}

// --- Settings: a fixed list of editable entries, index-ordered ---

/// A named model-size preset. Each sets the architecture at once.
#[derive(Clone, Copy)]
struct ModelTemplate {
    name: &'static str,
    d_model: usize,
    layers: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
}

const TEMPLATES: [ModelTemplate; 4] = [
    ModelTemplate { name: "tiny",   d_model: 128, layers: 2, heads: 4,  head_dim: 32, ffn: 512 },
    ModelTemplate { name: "small",  d_model: 256, layers: 4, heads: 8,  head_dim: 32, ffn: 512 },
    ModelTemplate { name: "medium", d_model: 384, layers: 6, heads: 12, head_dim: 32, ffn: 1024 },
    ModelTemplate { name: "large",  d_model: 512, layers: 6, heads: 16, head_dim: 32, ffn: 1024 },
];

fn entries_from_settings(s: &Settings) -> Vec<Entry> {
    let model = if s.d_model == 128 && s.layers == 2 { "tiny" }
        else if s.d_model == 384 && s.layers == 6 && s.heads == 12 { "medium" }
        else if s.d_model == 512 && s.layers == 6 && s.heads == 16 { "large" }
        else { "small" };
    let template_label = if s.model_template == "custom" { "custom".to_string() } else { model.to_string() };

    let order: [(String, String); 21] = [
        ("Model template".into(), template_label),
        ("Device".into(), s.device.clone()),
        ("Data file".into(), s.data_file.clone()),
        ("Steps".into(), s.steps.to_string()),
        ("Batch size".into(), s.batch_size.to_string()),
        ("Context (seq_len)".into(), s.seq_len.to_string()),
        ("Learning rate".into(), s.lr.to_string()),
        ("Tokenizer".into(), s.tokenizer.clone()),
        ("BPE merges".into(), s.num_merges.to_string()),
        ("Chunk length".into(), s.chunk_len.to_string()),
        ("d_model".into(), s.d_model.to_string()),
        ("Layers".into(), s.layers.to_string()),
        ("Heads".into(), s.heads.to_string()),
        ("Head dim".into(), s.head_dim.to_string()),
        ("FFN".into(), s.ffn.to_string()),
        ("Decay min".into(), s.decay_min.to_string()),
        ("Decay max".into(), s.decay_max.to_string()),
        ("Chat template".into(), s.chat_template.clone()),
        ("Temperature".into(), s.temperature.to_string()),
        ("Top-k".into(), s.top_k.to_string()),
        ("Answer tokens".into(), s.tokens.to_string()),
    ];
    order
        .into_iter()
        .map(|(label, value)| Entry { label, value })
        .collect()
}

fn apply_settings(entries: &[Entry], base: &Settings) -> Settings {
    let g = |i: usize| entries.get(i).map(|e| e.value.clone()).unwrap_or_default();
    Settings {
        model_template: g(0),
        device: g(1),
        data_file: g(2),
        steps: g(3).parse().unwrap_or(base.steps),
        batch_size: g(4).parse().unwrap_or(base.batch_size),
        seq_len: g(5).parse().unwrap_or(base.seq_len),
        lr: g(6).parse().unwrap_or(base.lr),
        tokenizer: g(7),
        num_merges: g(8).parse().unwrap_or(base.num_merges),
        chunk_len: g(9).parse().unwrap_or(base.chunk_len),
        d_model: g(10).parse().unwrap_or(base.d_model),
        layers: g(11).parse().unwrap_or(base.layers),
        heads: g(12).parse().unwrap_or(base.heads),
        head_dim: g(13).parse().unwrap_or(base.head_dim),
        ffn: g(14).parse().unwrap_or(base.ffn),
        decay_min: g(15).parse().unwrap_or(base.decay_min),
        decay_max: g(16).parse().unwrap_or(base.decay_max),
        chat_template: g(17),
        temperature: g(18).parse().unwrap_or(base.temperature),
        top_k: g(19).parse().unwrap_or(base.top_k),
        tokens: g(20).parse().unwrap_or(base.tokens),
    }
}

fn banner_text() -> String {
    format!(
        "
      ##  ##  ####  ####   ###### ###### ##  ##  ###
      ##  ## #    # #   #    ##   #      ##  ## #    #
      ##  ## #    # ####     ##   #####    ##   ######
       ## ## #    # #  #     ##   #      ##  ## #    #
        ###   ####  #   #    ##   ###### ##  ## #    #
                           v{}
      a tiny RetNet model · CPU / GPU · Rust
",
        env!("CARGO_PKG_VERSION")
    )
}

fn config_from_checkpoint(ckpt: &Path) -> Config {
    let (_, cfg_path) = train::resolve_checkpoint_paths(ckpt);
    std::fs::read_to_string(&cfg_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(Config::larger)
}

pub fn run() -> Result<()> {
    // Create `data/` and a default `settings.json` on first run, so a fresh
    // install (or an unchanged release zip) works right away.
    crate::settings::init_workspace()?;

    enable_raw_mode()?;
    let mut stdout: Stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let result = event_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    let _ = terminal.flush();
    result
}

fn event_loop(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> Result<()> {
    loop {
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                app.on_key(key.code, ctrl);
            }
        }
        terminal.draw(|f| app.draw(f))?;
        if app.quit {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn render(app: &mut App) -> bool {
        let backend = TestBackend::new(90, 28);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .any(|c| !c.symbol().is_empty() && c.symbol().trim() != "")
    }

    #[test]
    fn menu_renders() {
        let mut app = App::new();
        app.screen = Screen::Menu;
        assert!(render(&mut app));
    }

    #[test]
    fn settings_renders() {
        let mut app = App::new();
        app.screen = Screen::Settings;
        assert!(render(&mut app));
    }

    #[test]
    fn train_form_renders() {
        let mut app = App::new();
        app.screen = Screen::TrainForm;
        assert!(render(&mut app));
    }

    #[test]
    fn train_run_renders() {
        let mut app = App::new();
        app.screen = Screen::TrainRun;
        {
            let mut g = app.train_progress.lock().unwrap();
            g.step = 50;
            g.total = 100;
            g.loss = 2.5;
            g.tok_s = 8000.0;
        }
        assert!(render(&mut app));
    }

    #[test]
    fn datasets_renders() {
        let mut app = App::new();
        app.screen = Screen::Datasets;
        assert!(render(&mut app));
    }

    #[test]
    fn chat_renders_and_scrolls() {
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.chat_input = "hello".to_string();
        app.submit_chat();
        assert_eq!(app.chat_history.len(), 1);
        assert!(app.chat_scroll_pending);
        assert!(render(&mut app));
    }

    #[test]
    fn about_renders() {
        let mut app = App::new();
        app.screen = Screen::About;
        assert!(render(&mut app));
    }

    #[test]
    fn settings_roundtrip() {
        let mut s = Settings::default();
        s.num_merges = 999;
        let entries = entries_from_settings(&s);
        let back = apply_settings(&entries, &Settings::default());
        assert_eq!(back.num_merges, 999);
    }
}
