use ratatui::Frame;
use ratatui::layout::Constraint;
use ratatui::layout::Direction;
use ratatui::layout::Layout;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::widgets::Block;
use ratatui::widgets::Borders;
use ratatui::widgets::List;
use ratatui::widgets::ListItem;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Tabs;
use ratatui::widgets::Wrap;

use super::App;
use super::Page;
use super::mask_secret;

pub(super) fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(3),
        ])
        .split(area);

    render_header(frame, app, layout[0]);
    render_tabs(frame, app, layout[1]);
    match app.page {
        Page::Chat => render_chat(frame, app, layout[2]),
        Page::Settings => render_settings(frame, app, layout[2]),
    }
    render_status(frame, app, layout[3]);
}

fn render_header(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let width = area.width.max(24) as usize;
    let header = app.header_lines(width.saturating_sub(2)).join("\n");
    let widget = Paragraph::new(header)
        .block(Block::default().borders(Borders::ALL).title("CTOX"))
        .wrap(Wrap { trim: false });
    frame.render_widget(widget, area);
}

fn render_tabs(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let selected = match app.page {
        Page::Chat => 0,
        Page::Settings => 1,
    };
    let titles = ["Chat", "Settings"]
        .into_iter()
        .map(Line::from)
        .collect::<Vec<_>>();
    let widget = Tabs::new(titles)
        .select(selected)
        .block(Block::default().borders(Borders::ALL).title("Workspace"))
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    frame.render_widget(widget, area);
}

fn render_chat(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(5)])
        .split(area);

    let mut transcript = String::new();
    for message in app.chat_messages.iter().rev().take(24).collect::<Vec<_>>().into_iter().rev() {
        let role = if message.role.eq_ignore_ascii_case("assistant") {
            "assistant"
        } else {
            "user"
        };
        transcript.push_str(role);
        transcript.push_str("> ");
        transcript.push_str(&message.content);
        transcript.push_str("\n\n");
    }
    if app.request_in_flight {
        transcript.push_str("assistant> thinking...");
    }

    let transcript_widget = Paragraph::new(transcript)
        .block(Block::default().borders(Borders::ALL).title("Conversation"))
        .wrap(Wrap { trim: false });
    frame.render_widget(transcript_widget, layout[0]);

    let prompt_title = if app.request_in_flight {
        "Composer (request running)"
    } else {
        "Composer"
    };
    let composer = Paragraph::new(app.chat_input.as_str())
        .block(Block::default().borders(Borders::ALL).title(prompt_title))
        .wrap(Wrap { trim: false });
    frame.render_widget(composer, layout[1]);
}

fn render_settings(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(5)])
        .split(area);

    let items = app
        .settings_items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let rendered_value = if item.secret && !item.value.trim().is_empty() {
                mask_secret(&item.value)
            } else if item.value.trim().is_empty() {
                "(empty)".to_string()
            } else {
                item.value.clone()
            };
            let base = format!("{:18} {}", item.label, rendered_value);
            if index == app.settings_selected {
                ListItem::new(base).style(
                    Style::default()
                        .bg(Color::Cyan)
                        .fg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ListItem::new(base)
            }
        })
        .collect::<Vec<_>>();
    let title = if app.settings_dirty {
        "Settings (unsaved)"
    } else {
        "Settings"
    };
    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(list, layout[0]);

    let help = app
        .settings_items
        .get(app.settings_selected)
        .map(|item| {
            let mut text = format!("Help: {}", item.help);
            if !item.choices.is_empty() {
                text.push_str("\nChoices: ");
                text.push_str(&item.choices.join(", "));
            }
            text
        })
        .unwrap_or_else(|| "No setting selected.".to_string());
    let help_widget = Paragraph::new(help)
        .block(Block::default().borders(Borders::ALL).title("Details"))
        .wrap(Wrap { trim: false });
    frame.render_widget(help_widget, layout[1]);
}

fn render_status(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let spinner = if app.request_in_flight {
        ["|", "/", "-", "\\"][app.spinner_phase]
    } else {
        " "
    };
    let help = match app.page {
        Page::Chat => "Enter send, Tab settings, Ctrl-C quit",
        Page::Settings => "Up/Down select, Left/Right cycle, type to edit, Ctrl-S save, Tab chat",
    };
    let body = format!("{spinner} {}\n{help}", app.status_line);
    let widget = Paragraph::new(body)
        .block(Block::default().borders(Borders::ALL).title("Status"))
        .wrap(Wrap { trim: false });
    frame.render_widget(widget, area);
}
