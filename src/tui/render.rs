use ratatui::layout::Constraint;
use ratatui::layout::Direction;
use ratatui::layout::Layout;
use ratatui::prelude::Alignment;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Block;
use ratatui::widgets::BorderType;
use ratatui::widgets::Borders;
use ratatui::widgets::List;
use ratatui::widgets::ListItem;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Tabs;
use ratatui::widgets::Wrap;
use ratatui::Frame;

use crate::execution_baseline;

use super::compact_model_name;
use super::App;
use super::Page;
use super::SettingsView;

pub(super) fn draw(frame: &mut Frame, app: &App) {
    frame.render_widget(
        Block::default().style(Style::default().bg(Color::Black)),
        frame.area(),
    );
    let area = frame.area();
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(1),
            Constraint::Min(8),
            Constraint::Length(0),
        ])
        .split(area);

    render_header(frame, app, layout[0]);
    render_tabs(frame, app, layout[1]);
    match app.page {
        Page::Chat => render_chat(frame, app, layout[2]),
        Page::Skills => render_skills(frame, app, layout[2]),
        Page::Settings => render_settings(frame, app, layout[2]),
    }
    render_status(frame, app, layout[3]);
}

fn render_header(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let width = area.width.max(24) as usize;
    let header = header_lines(app, width.saturating_sub(4));
    let widget = Paragraph::new(header)
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray))
                .style(Style::default().bg(Color::Rgb(8, 8, 8)))
                .title(Span::styled(
                    " CTOX ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::LightCyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .title_alignment(Alignment::Left),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(widget, area);
}

fn render_tabs(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let selected = match app.page {
        Page::Chat => 0,
        Page::Skills => 1,
        Page::Settings => 2,
    };
    let titles = ["Chat", "Skills", "Settings"]
        .into_iter()
        .map(Line::from)
        .collect::<Vec<_>>();
    let widget = Tabs::new(titles)
        .select(selected)
        .divider(" ")
        .padding("", "")
        .style(Style::default().fg(Color::DarkGray).bg(Color::Rgb(8, 8, 8)))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(widget, area);
}

fn render_chat(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    if area.width < 96 {
        render_chat_narrow(frame, app, area);
        return;
    }
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
        .split(area);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(10), Constraint::Length(5)])
        .split(body[0]);
    let right = body[1];

    let turn_widget = Paragraph::new(turn_summary_lines(
        app,
        left[0].width.saturating_sub(4) as usize,
        left[0].height.saturating_sub(2) as usize,
    ))
    .block(
        pane_block().borders(Borders::TOP).title(Span::styled(
            " turn ",
            Style::default()
                .fg(Color::LightMagenta)
                .add_modifier(Modifier::BOLD),
        )),
    )
    .style(Style::default().fg(Color::White))
    .wrap(Wrap { trim: false });
    frame.render_widget(turn_widget, left[0]);

    let transcript_widget = Paragraph::new(render_transcript_lines(
        app,
        left[1].width.saturating_sub(4) as usize,
        left[1].height.saturating_sub(2) as usize,
    ))
        .block(
            pane_block().borders(Borders::TOP).title(Span::styled(
                " chat ",
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            )),
        )
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: false });
    frame.render_widget(transcript_widget, left[1]);

    let sidebar_widget = Paragraph::new(chat_sidebar_lines(
        app,
        right.width.saturating_sub(4) as usize,
        right.height.saturating_sub(2) as usize,
    ))
    .block(sidebar_block())
    .style(Style::default().fg(Color::Gray))
    .wrap(Wrap { trim: false });
    frame.render_widget(sidebar_widget, right);

    let composer_text = if app.chat_input.trim().is_empty() {
        if app.request_in_flight {
            "Type while CTOX is busy. Enter queues the draft.".to_string()
        } else {
            "Type your next instruction.".to_string()
        }
    } else {
        app.chat_input.clone()
    };
    let composer = Paragraph::new(composer_text)
        .alignment(Alignment::Left)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(Span::styled(
                    if app.request_in_flight {
                        " queued draft "
                    } else {
                        " compose "
                    },
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ))
                .border_style(Style::default().fg(Color::DarkGray))
                .style(Style::default().bg(Color::Rgb(20, 20, 20))),
        )
        .style(if app.chat_input.trim().is_empty() {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::White)
        })
        .wrap(Wrap { trim: false });
    frame.render_widget(composer, left[2]);
}

fn render_skills(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    if area.width < 96 {
        render_skills_narrow(frame, app, area);
        return;
    }
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(36), Constraint::Percentage(64)])
        .split(area);

    let list_items = skill_list_items(app, body[0].width.saturating_sub(4) as usize);
    frame.render_widget(
        List::new(list_items).block(
            pane_block().borders(Borders::TOP).title(Span::styled(
                " skills ",
                Style::default()
                    .fg(Color::LightYellow)
                    .add_modifier(Modifier::BOLD),
            )),
        ),
        body[0],
    );

    let details = Paragraph::new(skill_details_lines(
        app,
        body[1].width.saturating_sub(4) as usize,
        body[1].height.saturating_sub(2) as usize,
    ))
    .block(
        sidebar_block().borders(Borders::TOP).title(Span::styled(
            " selected skill ",
            Style::default()
                .fg(Color::LightBlue)
                .add_modifier(Modifier::BOLD),
        )),
    )
    .style(Style::default().fg(Color::White))
    .wrap(Wrap { trim: false });
    frame.render_widget(details, body[1]);
}

fn render_settings(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    if area.width < 96 {
        render_settings_narrow(frame, app, area);
        return;
    }
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(area);

    let visible_indices = app.visible_setting_indices();
    let max_rows = body[0].height.saturating_sub(2) as usize;
    let window_indices = settings_window_indices(app, &visible_indices, max_rows);
    let items = window_indices
        .into_iter()
        .filter_map(|idx| app.settings_items.get(idx).map(|item| (idx, item)))
        .map(|(index, item)| {
            let rendered_value = app.rendered_setting_value(item);
            let base = format!("{:18} {}", item.label, truncate_line(&rendered_value, 44));
            let row_style = setting_row_style(item.key, item.value.trim());
            if index == app.settings_selected {
                ListItem::new(base).style(
                    row_style
                        .bg(Color::LightCyan)
                        .fg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ListItem::new(base).style(row_style)
            }
        })
        .collect::<Vec<_>>();
    let title = if app.settings_dirty {
        " settings * "
    } else {
        " settings "
    };
    let list = List::new(items).block(
        pane_block().borders(Borders::TOP).title(Span::styled(
            title,
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        )),
    );
    frame.render_widget(list, body[0]);

    let help_widget = Paragraph::new(settings_snapshot_text(
        app,
        body[1].width.saturating_sub(4) as usize,
        body[1].height.saturating_sub(2) as usize,
    ))
    .block(
        sidebar_block().borders(Borders::TOP).title(Span::styled(
            if app.header.estimate_mode {
                " estimate "
            } else {
                " live "
            },
            Style::default()
                .fg(Color::LightBlue)
                .add_modifier(Modifier::BOLD),
        )),
    )
    .style(Style::default().fg(Color::Gray))
    .wrap(Wrap { trim: false });
    frame.render_widget(help_widget, body[1]);
}

fn render_skills_narrow(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(38), Constraint::Min(10)])
        .split(area);
    frame.render_widget(
        List::new(skill_list_items(
            app,
            layout[0].width.saturating_sub(4) as usize,
        ))
        .block(
            pane_block().borders(Borders::TOP).title(Span::styled(
                " skills ",
                Style::default()
                    .fg(Color::LightYellow)
                    .add_modifier(Modifier::BOLD),
            )),
        ),
        layout[0],
    );
    frame.render_widget(
        Paragraph::new(skill_details_lines(
            app,
            layout[1].width.saturating_sub(4) as usize,
            layout[1].height.saturating_sub(2) as usize,
        ))
        .block(
            sidebar_block().borders(Borders::TOP).title(Span::styled(
                " details ",
                Style::default()
                    .fg(Color::LightBlue)
                    .add_modifier(Modifier::BOLD),
            )),
        )
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: false }),
        layout[1],
    );
}

fn render_status(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let _ = (frame, app, area);
}

fn render_chat_narrow(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Percentage(30),
            Constraint::Percentage(24),
            Constraint::Min(5),
        ])
        .split(area);

    let turn_widget = Paragraph::new(turn_summary_lines(
        app,
        layout[0].width.saturating_sub(4) as usize,
        layout[0].height.saturating_sub(2) as usize,
    ))
    .block(
        pane_block().borders(Borders::TOP).title(Span::styled(
            " turn ",
            Style::default()
                .fg(Color::LightMagenta)
                .add_modifier(Modifier::BOLD),
        )),
    )
    .style(Style::default().fg(Color::White))
    .wrap(Wrap { trim: false });
    frame.render_widget(turn_widget, layout[0]);

    let transcript_widget = Paragraph::new(render_transcript_lines(
        app,
        layout[1].width.saturating_sub(4) as usize,
        layout[1].height.saturating_sub(2) as usize,
    ))
        .block(
            pane_block().borders(Borders::TOP).title(Span::styled(
                " chat ",
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            )),
        )
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: false });
    frame.render_widget(transcript_widget, layout[1]);

    let feed = Paragraph::new(
        activity_lines(app, layout[2].width.saturating_sub(4) as usize, 2, true).join("\n"),
    )
    .block(
        sidebar_block().borders(Borders::TOP).title(Span::styled(
            " live ",
            Style::default()
                .fg(Color::LightBlue)
                .add_modifier(Modifier::BOLD),
        )),
    )
    .style(Style::default().fg(Color::Gray))
    .wrap(Wrap { trim: false });
    frame.render_widget(feed, layout[2]);

    let composer_text = if app.chat_input.trim().is_empty() {
        if app.request_in_flight {
            "Type while CTOX is busy. Enter queues."
        } else {
            "Type your next instruction."
        }
    } else {
        app.chat_input.as_str()
    };
    let composer = Paragraph::new(composer_text)
        .alignment(Alignment::Left)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(Span::styled(
                    if app.request_in_flight {
                        " queued draft "
                    } else {
                        " compose "
                    },
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ))
                .border_style(Style::default().fg(Color::DarkGray))
                .style(Style::default().bg(Color::Rgb(20, 20, 20))),
        )
        .style(if app.chat_input.trim().is_empty() {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::White)
        })
        .wrap(Wrap { trim: false });
    frame.render_widget(composer, layout[3]);
}

fn render_settings_narrow(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(8)])
        .split(area);
    let visible_indices = app.visible_setting_indices();
    let max_rows = layout[0].height.saturating_sub(2) as usize;
    let window_indices = settings_window_indices(app, &visible_indices, max_rows);
    let items = window_indices
        .into_iter()
        .filter_map(|idx| app.settings_items.get(idx).map(|item| (idx, item)))
        .map(|(index, item)| {
            let rendered_value = app.rendered_setting_value(item);
            let base = format!("{:16} {}", item.label, truncate_line(&rendered_value, 32));
            let row_style = setting_row_style(item.key, item.value.trim());
            if index == app.settings_selected {
                ListItem::new(base).style(
                    row_style
                        .bg(Color::LightCyan)
                        .fg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ListItem::new(base).style(row_style)
            }
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(items).block(
            pane_block().borders(Borders::TOP).title(Span::styled(
                " settings ",
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            )),
        ),
        layout[0],
    );
    frame.render_widget(
        Paragraph::new(settings_snapshot_text(
            app,
            area.width.saturating_sub(6) as usize,
            layout[1].height.saturating_sub(2) as usize,
        ))
        .block(
            sidebar_block().borders(Borders::TOP).title(Span::styled(
                if app.header.estimate_mode {
                    " estimate "
                } else {
                    " live "
                },
                Style::default()
                    .fg(Color::LightBlue)
                    .add_modifier(Modifier::BOLD),
            )),
        )
        .style(Style::default().fg(Color::Gray))
        .wrap(Wrap { trim: false }),
        layout[1],
    );
}

fn render_transcript_lines(app: &App, width: usize, height: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for message in app
        .chat_messages
        .iter()
        .rev()
        .take(18)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let (label, badge_color, body_color) = if message.role.eq_ignore_ascii_case("assistant") {
            (" CTOX ", Color::LightGreen, Color::White)
        } else {
            (" YOU ", Color::LightCyan, Color::Gray)
        };
        lines.push(truncate_line_spans(
            vec![
                Span::styled(
                    label.to_string(),
                    Style::default()
                        .fg(Color::Black)
                        .bg(badge_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(
                    truncate_line(message.role.as_str(), width.saturating_sub(8)),
                    Style::default().fg(Color::DarkGray),
                ),
            ],
            width,
        ));
        for chunk in wrap_text_lines(&message.content, width.saturating_sub(2)) {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(chunk, Style::default().fg(body_color)),
            ]));
        }
        lines.push(Line::from(String::new()));
    }
    if app.request_in_flight {
        lines.push(truncate_line_spans(
            vec![
                Span::styled(
                    " WORKING ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(
                    app.service_status
                        .active_source_label
                        .clone()
                        .unwrap_or_else(|| "turn".to_string()),
                    Style::default().fg(Color::Yellow),
                ),
            ],
            width,
        ));
    }
    if lines.len() > height {
        lines = lines.split_off(lines.len().saturating_sub(height));
    }
    lines
}

fn settings_window_indices(app: &App, visible_indices: &[usize], max_rows: usize) -> Vec<usize> {
    if visible_indices.len() <= max_rows.max(1) {
        return visible_indices.to_vec();
    }
    let selected_pos = visible_indices
        .iter()
        .position(|idx| *idx == app.settings_selected)
        .unwrap_or(0);
    let window = max_rows.max(1);
    let max_start = visible_indices.len().saturating_sub(window);
    let start = selected_pos.min(max_start);
    visible_indices[start..start + window].to_vec()
}

fn skill_list_items(app: &App, width: usize) -> Vec<ListItem<'static>> {
    if app.skill_catalog.is_empty() {
        return vec![ListItem::new("No skills discovered yet.")
            .style(Style::default().fg(Color::DarkGray))];
    }
    let max_rows = app.skill_catalog.len();
    let window = skill_window_indices(app, max_rows.min(64));
    window
        .into_iter()
        .filter_map(|index| app.skill_catalog.get(index).map(|entry| (index, entry)))
        .map(|(index, entry)| {
            let row = format!(
                "{:18} {}",
                truncate_line(&entry.name, 18),
                truncate_line(&entry.source, width.saturating_sub(20))
            );
            let base_style = if entry.source.contains("system") {
                Style::default().fg(Color::LightCyan)
            } else {
                Style::default().fg(Color::White)
            };
            if index == app.skills_selected {
                ListItem::new(row).style(
                    base_style
                        .bg(Color::LightYellow)
                        .fg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ListItem::new(row).style(base_style)
            }
        })
        .collect()
}

fn skill_window_indices(app: &App, max_rows: usize) -> Vec<usize> {
    let total = app.skill_catalog.len();
    if total <= max_rows.max(1) {
        return (0..total).collect();
    }
    let window = max_rows.max(1);
    let selected = app.skills_selected.min(total.saturating_sub(1));
    let half = window / 2;
    let mut start = selected.saturating_sub(half);
    let max_start = total.saturating_sub(window);
    if start > max_start {
        start = max_start;
    }
    (start..start + window).collect()
}

fn skill_details_lines(app: &App, width: usize, height: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let Some(entry) = app.skill_catalog.get(app.skills_selected) else {
        return vec![Line::from("No skills found.")];
    };
    lines.push(Line::from(vec![
        Span::styled(
            entry.name.clone(),
            Style::default()
                .fg(Color::LightYellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(entry.source.clone(), Style::default().fg(Color::LightCyan)),
    ]));
    lines.push(Line::from(format!(
        "path {}",
        truncate_line(&entry.skill_path.to_string_lossy(), width.saturating_sub(5))
    )));
    lines.push(Line::from(String::new()));
    lines.push(section_title("summary", Color::LightGreen, width));
    for chunk in wrap_text_lines(&entry.description, width) {
        lines.push(Line::from(chunk));
    }
    lines.push(Line::from(String::new()));
    lines.push(section_title("helper tools", Color::LightMagenta, width));
    if entry.helper_tools.is_empty() {
        lines.push(Line::from("No scripts/ helper tools detected."));
    } else {
        for tool in entry.helper_tools.iter().take(10) {
            lines.push(Line::from(format!("• {}", truncate_line(tool, width.saturating_sub(2)))));
        }
    }
    lines.push(Line::from(String::new()));
    lines.push(section_title("resources", Color::LightBlue, width));
    if entry.resources.is_empty() {
        lines.push(Line::from("No extra resources detected."));
    } else {
        for resource in entry.resources.iter().take(10) {
            for chunk in wrap_text_lines(resource, width.saturating_sub(2)) {
                lines.push(Line::from(format!("• {chunk}")));
            }
        }
    }
    lines.push(Line::from(String::new()));
    lines.push(Line::from("Up/Down select  R reload  Tab next page"));
    if lines.len() > height {
        lines.truncate(height);
    }
    lines
}

fn activity_lines(
    app: &App,
    width: usize,
    channel_limit: usize,
    include_queue: bool,
) -> Vec<String> {
    let mut lines = if app.activity_log.is_empty() {
        vec!["• Waiting for the next CTOX event.".to_string()]
    } else {
        app.activity_log
            .iter()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|line| format!("• {}", truncate_line(line, width)))
            .collect::<Vec<_>>()
    };
    if include_queue {
        if app.draft_queue.is_empty() {
            if !app.service_status.pending_previews.is_empty() {
                lines.push(format!(
                    "queue  {} waiting",
                    app.service_status.pending_count
                ));
                for preview in app.service_status.pending_previews.iter().take(3) {
                    lines.push(format!("• {}", truncate_line(preview, width)));
                }
            } else if app.service_status.pending_count > 0 {
                lines.push(format!(
                    "queue  {} waiting",
                    app.service_status.pending_count
                ));
            } else {
                lines.push("queue  clear".to_string());
            }
        } else {
            lines.push(format!("queue  {} local", app.draft_queue.len()));
            for draft in app.draft_queue.iter().take(2) {
                lines.push(format!("• {}", truncate_line(draft, width)));
            }
        }
    }
    lines.push("inbox".to_string());
    if app.communication_feed.is_empty() {
        lines.push("• No channel traffic yet.".to_string());
    } else {
        for item in app.communication_feed.iter().take(channel_limit) {
            let direction = if item.direction == "inbound" {
                "in"
            } else {
                "out"
            };
            let source = if !item.sender_display.trim().is_empty() {
                item.sender_display.trim()
            } else {
                item.sender_address.trim()
            };
            let text = if item.preview.trim().is_empty() {
                item.subject.as_str()
            } else {
                item.preview.as_str()
            };
            lines.push(format!(
                "• {} {} {}",
                item.channel,
                direction,
                truncate_line(
                    &format!("{source}: {text}"),
                    width.saturating_sub(item.channel.len() + 4)
                )
            ));
        }
    }
    lines
}

fn pane_block() -> Block<'static> {
    Block::default()
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(Color::DarkGray))
        .style(Style::default().bg(Color::Rgb(10, 10, 10)))
}

fn sidebar_block() -> Block<'static> {
    Block::default()
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(Color::DarkGray))
        .style(Style::default().bg(Color::Rgb(18, 18, 18)))
}

fn sidebar_fill() -> Block<'static> {
    Block::default().style(Style::default().bg(Color::Rgb(18, 18, 18)))
}

fn sidebar_footer_lines(app: &App) -> Vec<Line<'static>> {
    let frames = ["⠁", "⠂", "⠄", "⠂"];
    let spinner = frames[app.spinner_phase % frames.len()];
    let status = if app.request_in_flight {
        "working"
    } else if app.service_status.running {
        "ready"
    } else {
        "stopped"
    };
    vec![
        Line::from(vec![
            Span::styled(format!("{spinner} "), Style::default().fg(Color::LightBlue)),
            Span::styled(status, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Esc", Style::default().fg(Color::White)),
            Span::raw(" interrupt  "),
            Span::styled("Tab", Style::default().fg(Color::White)),
            Span::raw(" next page"),
        ]),
        Line::from(vec![
            Span::styled("Ctrl-C", Style::default().fg(Color::White)),
            Span::raw(" quit"),
        ]),
    ]
}

fn chat_sidebar_lines(app: &App, width: usize, height: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let footer = sidebar_footer_lines(app);
    let footer_reserved = footer.len();
    let queue_reserved = 4usize;
    let live_budget = height
        .saturating_sub(footer_reserved + queue_reserved)
        .max(4);
    lines.push(section_title("live", Color::LightBlue, width));
    for line in activity_lines(app, width, 2, false)
        .into_iter()
        .take(live_budget.saturating_sub(1))
    {
        lines.push(Line::from(truncate_line(&line, width)));
    }
    lines.push(Line::from(String::new()));

    let queue_title = if app.draft_queue.is_empty() {
        "queue".to_string()
    } else {
        format!("queue {}", app.draft_queue.len())
    };
    lines.push(section_title(&queue_title, Color::LightYellow, width));
    for line in queue_lines(app, width) {
        lines.push(Line::from(truncate_line(&line, width)));
    }

    while lines.len() + footer_reserved < height {
        lines.push(Line::from(String::new()));
    }
    lines.extend(footer);
    lines.into_iter().take(height.max(1)).collect()
}

fn queue_lines(app: &App, width: usize) -> Vec<String> {
    if app.draft_queue.is_empty() {
        if !app.service_status.pending_previews.is_empty() {
            let mut lines = vec![format!("{} waiting", app.service_status.pending_count)];
            for preview in app.service_status.pending_previews.iter().take(3) {
                lines.push(format!(
                    "• {}",
                    truncate_line(preview, width.saturating_sub(2))
                ));
            }
            return lines;
        }
        if app.service_status.pending_count > 0 {
            return vec![format!(
                "{} server-side prompt(s) waiting.",
                app.service_status.pending_count
            )];
        }
        return vec!["No queued drafts.".to_string()];
    }

    app.draft_queue
        .iter()
        .enumerate()
        .map(|(idx, draft)| {
            format!(
                "{}. {}",
                idx + 1,
                truncate_line(draft, width.saturating_sub(3))
            )
        })
        .collect()
}

fn section_title(title: &str, color: Color, width: usize) -> Line<'static> {
    let label = format!(" {title} ");
    let dash_count = width.saturating_sub(label.chars().count()).max(1);
    let mut spans = vec![Span::styled(
        label,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )];
    spans.push(Span::styled(
        "─".repeat(dash_count),
        Style::default().fg(Color::DarkGray),
    ));
    truncate_line_spans(spans, width)
}

fn turn_summary_lines(app: &App, width: usize, height: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let (status_text, status_color) = if !app.service_status.running {
        ("stopped", Color::Red)
    } else if app.service_status.busy {
        ("working", Color::Yellow)
    } else {
        ("ready", Color::Green)
    };
    lines.push(truncate_line_spans(
        vec![
            Span::raw("state "),
            Span::styled(
                status_text.to_string(),
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                app.service_status
                    .active_source_label
                    .clone()
                    .unwrap_or_else(|| "idle".to_string()),
                Style::default().fg(Color::LightCyan),
            ),
        ],
        width,
    ));
    lines.push(Line::from(format!(
        "queue {} pending  drafts {}",
        app.service_status.pending_count,
        app.draft_queue.len()
    )));
    if let Some(error) = app.service_status.last_error.as_deref() {
        lines.push(Line::from(format!(
            "error {}",
            truncate_line(error, width.saturating_sub(6))
        )));
    } else if let Some(chars) = app.service_status.last_reply_chars {
        lines.push(Line::from(format!("last reply {chars} chars")));
    } else {
        lines.push(Line::from("last reply -"));
    }
    if let Some(completed) = app.service_status.last_completed_at.as_deref() {
        lines.push(Line::from(format!(
            "completed {}",
            truncate_line(completed, width.saturating_sub(10))
        )));
    }
    for event in app.service_status.recent_events.iter().rev().take(3).rev() {
        lines.push(Line::from(format!(
            "• {}",
            truncate_line(event, width.saturating_sub(2))
        )));
    }
    lines.truncate(height);
    lines
}

fn wrap_text_lines(value: &str, max_chars: usize) -> Vec<String> {
    if max_chars == 0 {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    for paragraph in value.lines() {
        let words = paragraph.split_whitespace().collect::<Vec<_>>();
        if words.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current = String::new();
        for word in words {
            let proposed_len = if current.is_empty() {
                word.chars().count()
            } else {
                current.chars().count() + 1 + word.chars().count()
            };
            if proposed_len > max_chars && !current.is_empty() {
                lines.push(current);
                current = word.to_string();
            } else if current.is_empty() {
                current = word.to_string();
            } else {
                current.push(' ');
                current.push_str(word);
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn value_for_key(app: &App, key: &str) -> String {
    app.settings_items
        .iter()
        .find(|item| item.key == key)
        .map(|item| {
            let value = item.value.trim();
            if value.is_empty() {
                "-".to_string()
            } else {
                truncate_line(value, 30)
            }
        })
        .unwrap_or_else(|| "-".to_string())
}

fn settings_snapshot_text(app: &App, width: usize, height: usize) -> String {
    let mode = if app.header.estimate_mode {
        "estimate"
    } else {
        "live"
    };
    let avg_tps = app
        .header
        .avg_tokens_per_second
        .map(|value| format!("{value:.0} tok/s"))
        .unwrap_or_else(|| "- tok/s".to_string());
    let base_model = value_for_key(app, "CTOX_CHAT_MODEL_BASE");
    let active_model = truncate_line(&app.header.model, width.saturating_sub(9));
    let boost_model = value_for_key(app, "CTOX_CHAT_MODEL_BOOST");
    let boost_status = if app.header.boost_active {
        if let Some(remaining_seconds) = app.header.boost_remaining_seconds {
            format!("active {}m", (remaining_seconds + 59) / 60)
        } else {
            "active".to_string()
        }
    } else if boost_model != "-" {
        "idle".to_string()
    } else {
        "-".to_string()
    };
    let boost_reason = app
        .header
        .boost_reason
        .as_deref()
        .map(|value| truncate_line(value, width.saturating_sub(9)))
        .unwrap_or_else(|| "-".to_string());
    let chat_source = value_for_key(app, "CTOX_CHAT_SOURCE");
    let channel = value_for_key(app, "CTOX_OWNER_PREFERRED_CHANNEL");
    if channel == "email" {
        let protocol = value_for_key(app, "CTO_EMAIL_PROVIDER");
        let mut lines = vec![
            format!("mode     {mode}"),
            "chat".to_string(),
            format!("base     {}", truncate_line(&base_model, width.saturating_sub(9))),
            format!("active   {active_model}"),
            format!("boost    {}", truncate_line(&boost_status, width.saturating_sub(9))),
            format!("why      {}", truncate_line(&boost_reason, width.saturating_sub(9))),
            String::new(),
            format!(
                "owner    {}",
                truncate_line(
                    &value_for_key(app, "CTOX_OWNER_NAME"),
                    width.saturating_sub(9)
                )
            ),
            format!(
                "owner@   {}",
                truncate_line(
                    &value_for_key(app, "CTOX_OWNER_EMAIL_ADDRESS"),
                    width.saturating_sub(9)
                )
            ),
            "channel  email".to_string(),
            String::new(),
            "email".to_string(),
            format!(
                "address  {}",
                truncate_line(
                    &value_for_key(app, "CTO_EMAIL_ADDRESS"),
                    width.saturating_sub(9)
                )
            ),
            format!(
                "proto    {}",
                truncate_line(&protocol, width.saturating_sub(9))
            ),
            "choices  imap | graph | ews".to_string(),
        ];
        match protocol.as_str() {
            "graph" => lines.push(format!(
                "user     {}",
                truncate_line(
                    &value_for_key(app, "CTO_EMAIL_GRAPH_USER"),
                    width.saturating_sub(9)
                )
            )),
            "ews" => {
                lines.push(format!(
                    "url      {}",
                    truncate_line(
                        &value_for_key(app, "CTO_EMAIL_EWS_URL"),
                        width.saturating_sub(9)
                    )
                ));
                lines.push(format!(
                    "auth     {}",
                    truncate_line(
                        &value_for_key(app, "CTO_EMAIL_EWS_AUTH_TYPE"),
                        width.saturating_sub(9)
                    )
                ));
                lines.push(format!(
                    "user     {}",
                    truncate_line(
                        &value_for_key(app, "CTO_EMAIL_EWS_USERNAME"),
                        width.saturating_sub(9)
                    )
                ));
            }
            _ => {
                lines.push(format!(
                    "imap     {}:{}",
                    value_for_key(app, "CTO_EMAIL_IMAP_HOST"),
                    value_for_key(app, "CTO_EMAIL_IMAP_PORT")
                ));
                lines.push(format!(
                    "smtp     {}:{}",
                    value_for_key(app, "CTO_EMAIL_SMTP_HOST"),
                    value_for_key(app, "CTO_EMAIL_SMTP_PORT")
                ));
            }
        }
        lines
            .into_iter()
            .take(height.max(1))
            .collect::<Vec<_>>()
            .join("\n")
    } else if channel == "jami" {
        let mut lines = vec![
            format!("mode     {mode}"),
            "chat".to_string(),
            format!("base     {}", truncate_line(&base_model, width.saturating_sub(9))),
            format!("active   {active_model}"),
            format!("boost    {}", truncate_line(&boost_status, width.saturating_sub(9))),
            format!("why      {}", truncate_line(&boost_reason, width.saturating_sub(9))),
            String::new(),
            format!(
                "owner    {}",
                truncate_line(
                    &value_for_key(app, "CTOX_OWNER_NAME"),
                    width.saturating_sub(9)
                )
            ),
            "channel  jami".to_string(),
            String::new(),
            "jami".to_string(),
            format!(
                "name     {}",
                truncate_line(
                    &value_for_key(app, "CTO_JAMI_PROFILE_NAME"),
                    width.saturating_sub(9)
                )
            ),
            format!(
                "account  {}",
                truncate_line(
                    &value_for_key(app, "CTO_JAMI_ACCOUNT_ID"),
                    width.saturating_sub(9)
                )
            ),
            String::new(),
        ];
        for qr in app.jami_qr_lines.iter() {
            lines.push(truncate_line(qr, width));
        }
        lines
            .into_iter()
            .take(height.max(1))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        let mut lines = vec![
            format!("mode     {mode}"),
            format!("source   {chat_source}"),
            format!("base     {}", truncate_line(&base_model, width.saturating_sub(9))),
            format!("active   {active_model}"),
            format!("boost    {}", truncate_line(&boost_status, width.saturating_sub(9))),
            format!("why      {}", truncate_line(&boost_reason, width.saturating_sub(9))),
        ];
        if chat_source == "api" {
            lines.push(format!(
                "provider {}",
                truncate_line(
                    &value_for_key(app, "CTOX_API_PROVIDER"),
                    width.saturating_sub(9)
                )
            ));
        } else {
            lines.push(format!(
                "preset   {}",
                truncate_line(
                    &value_for_key(app, "CTOX_CHAT_LOCAL_PRESET"),
                    width.saturating_sub(9)
                )
            ));
        }
        lines.push(format!(
            "owner    {}",
            truncate_line(
                &value_for_key(app, "CTOX_OWNER_NAME"),
                width.saturating_sub(9)
            )
        ));
        lines.push(format!(
            "owner@   {}",
            truncate_line(
                &value_for_key(app, "CTOX_OWNER_EMAIL_ADDRESS"),
                width.saturating_sub(9)
            )
        ));
        lines.push(format!(
            "channel  {}",
            truncate_line(&channel, width.saturating_sub(9))
        ));
        if channel == "email" {
            lines.push(String::new());
            lines.push("email".to_string());
            lines.push(format!(
                "address  {}",
                truncate_line(
                    &value_for_key(app, "CTO_EMAIL_ADDRESS"),
                    width.saturating_sub(9)
                )
            ));
            let protocol = value_for_key(app, "CTO_EMAIL_PROVIDER");
            lines.push(format!(
                "proto    {}",
                truncate_line(&protocol, width.saturating_sub(9))
            ));
            lines.push("choices  imap | graph | ews".to_string());
            match protocol.as_str() {
                "graph" => lines.push(format!(
                    "user     {}",
                    truncate_line(
                        &value_for_key(app, "CTO_EMAIL_GRAPH_USER"),
                        width.saturating_sub(9)
                    )
                )),
                "ews" => {
                    lines.push(format!(
                        "url      {}",
                        truncate_line(
                            &value_for_key(app, "CTO_EMAIL_EWS_URL"),
                            width.saturating_sub(9)
                        )
                    ));
                    lines.push(format!(
                        "auth     {}",
                        truncate_line(
                            &value_for_key(app, "CTO_EMAIL_EWS_AUTH_TYPE"),
                            width.saturating_sub(9)
                        )
                    ));
                    lines.push(format!(
                        "user     {}",
                        truncate_line(
                            &value_for_key(app, "CTO_EMAIL_EWS_USERNAME"),
                            width.saturating_sub(9)
                        )
                    ));
                }
                _ => {
                    lines.push(format!(
                        "imap     {}:{}",
                        value_for_key(app, "CTO_EMAIL_IMAP_HOST"),
                        value_for_key(app, "CTO_EMAIL_IMAP_PORT")
                    ));
                    lines.push(format!(
                        "smtp     {}:{}",
                        value_for_key(app, "CTO_EMAIL_SMTP_HOST"),
                        value_for_key(app, "CTO_EMAIL_SMTP_PORT")
                    ));
                }
            }
        } else if channel == "jami" {
            lines.push(String::new());
            lines.push("jami".to_string());
            lines.push(format!(
                "name     {}",
                truncate_line(
                    &value_for_key(app, "CTO_JAMI_PROFILE_NAME"),
                    width.saturating_sub(9)
                )
            ));
            lines.push(format!(
                "account  {}",
                truncate_line(
                    &value_for_key(app, "CTO_JAMI_ACCOUNT_ID"),
                    width.saturating_sub(9)
                )
            ));
            lines.push(String::new());
            for qr in app.jami_qr_lines.iter() {
                lines.push(truncate_line(qr, width));
            }
        }
        lines.push(format!(
            "embed    {}",
            truncate_line(
                &value_for_key(app, "CTOX_EMBEDDING_MODEL"),
                width.saturating_sub(9)
            )
        ));
        lines.push(format!(
            "stt      {}",
            truncate_line(
                &value_for_key(app, "CTOX_STT_MODEL"),
                width.saturating_sub(9)
            )
        ));
        lines.push(format!(
            "tts      {}",
            truncate_line(
                &value_for_key(app, "CTOX_TTS_MODEL"),
                width.saturating_sub(9)
            )
        ));
        lines.push(format!(
            "compact  {}%",
            value_for_key(app, "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT")
        ));
        lines.push(format!("speed    {avg_tps}"));
        if channel != "email" && channel != "jami" {
            if let Some(bundle) = &app.chat_preset_bundle {
                lines.push(String::new());
                lines.push("plan".to_string());
                lines.push(format!(
                    "active   {}",
                    truncate_line(bundle.selected_plan.preset.label(), width.saturating_sub(9))
                ));
                lines.push(format!(
                    "quant    {}",
                    truncate_line(&bundle.selected_plan.quantization, width.saturating_sub(9))
                ));
                lines.push(format!("ctx      {}", bundle.selected_plan.max_seq_len));
                lines.push(format!(
                    "backend  {}",
                    if bundle.selected_plan.disable_nccl {
                        "device-layers"
                    } else {
                        "nccl"
                    }
                ));
                lines.push(format!(
                    "gpus     {}",
                    truncate_line(
                        &bundle.selected_plan.cuda_visible_devices,
                        width.saturating_sub(9)
                    )
                ));
                if let Some(device_layers) = &bundle.selected_plan.device_layers {
                    lines.push(format!(
                        "layers   {}",
                        truncate_line(device_layers, width.saturating_sub(9))
                    ));
                }
                lines.push("presets".to_string());
                for plan in &bundle.plans {
                    lines.push(format!(
                        "• {} {} {}k {:.0} tok/s",
                        plan.preset.label(),
                        plan.quantization,
                        plan.max_seq_len / 1024,
                        plan.expected_tok_s
                    ));
                }
                lines.push("gpu budget".to_string());
                for allocation in bundle.selected_plan.gpu_allocations.iter().take(4) {
                    lines.push(format!(
                        "• GPU{} {}G = {}G desk + {}G aux + {}G chat",
                        allocation.gpu_index,
                        allocation.total_mb / 1024,
                        allocation.desktop_reserve_mb / 1024,
                        allocation.aux_reserve_mb / 1024,
                        allocation.chat_budget_mb / 1024,
                    ));
                }
            }
        }
        lines
            .into_iter()
            .take(height.max(1))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn setting_details_text(app: &App) -> String {
    let Some(item) = app.current_setting() else {
        return String::new();
    };
    let mut lines = vec![app.selected_setting_help()];
    if item.kind == super::SettingKind::Env {
        lines.push(String::new());
        lines.push(format!(
            "saved   {}",
            display_setting_value(item.saved_value.trim(), item.secret)
        ));
        if app.setting_is_dirty(item) {
            lines.push(format!(
                "draft   {}",
                display_setting_value(item.value.trim(), item.secret)
            ));
            lines.push("Enter saves this change.".to_string());
        } else {
            lines.push(format!(
                "active  {}",
                display_setting_value(item.value.trim(), item.secret)
            ));
        }
    }
    if app.settings_menu_open && !item.choices.is_empty() {
        lines.push(String::new());
        lines.push("choose with ↑ ↓ and Enter".to_string());
        for (index, choice) in item.choices.iter().enumerate() {
            let marker = if app.settings_menu_open && app.settings_menu_index == index {
                "›"
            } else if item.value.trim().eq_ignore_ascii_case(choice) {
                "•"
            } else {
                " "
            };
            lines.push(format!("{marker} {choice}"));
        }
    }
    for line in setting_detail_footer_lines(item) {
        lines.push(line.to_string());
    }
    lines.join("\n")
}

fn setting_details_lines(app: &App) -> Vec<Line<'static>> {
    let Some(item) = app.current_setting() else {
        return Vec::new();
    };
    let mut lines = vec![Line::from(app.selected_setting_help())];
    if item.kind == super::SettingKind::Env {
        lines.push(Line::from(String::new()));
        lines.push(Line::from(format!(
            "saved   {}",
            display_setting_value(item.saved_value.trim(), item.secret)
        )));
        if app.setting_is_dirty(item) {
            lines.push(Line::from(format!(
                "draft   {}",
                display_setting_value(item.value.trim(), item.secret)
            )));
            lines.push(Line::from("Enter saves this change.".to_string()));
        } else {
            lines.push(Line::from(format!(
                "active  {}",
                display_setting_value(item.value.trim(), item.secret)
            )));
        }
    }
    if app.settings_menu_open && !item.choices.is_empty() {
        lines.push(Line::from(String::new()));
        lines.push(Line::from("choose with ↑ ↓ and Enter".to_string()));
        for (index, choice) in item.choices.iter().enumerate() {
            let marker = if app.settings_menu_index == index {
                "›"
            } else {
                " "
            };
            let style = if app.settings_menu_index == index {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD)
            } else if item.value.trim().eq_ignore_ascii_case(choice) {
                Style::default().fg(Color::LightGreen)
            } else {
                Style::default().fg(Color::Gray)
            };
            lines.push(Line::from(vec![Span::styled(
                format!("{marker} {choice}"),
                style,
            )]));
        }
    }
    lines.extend(setting_detail_footer_lines(item));
    lines
}

fn setting_detail_footer_lines(item: &super::SettingItem) -> Vec<Line<'static>> {
    match item.key {
        "CTO_EMAIL_PROVIDER" => vec![
            Line::from(String::new()),
            Line::from("Protocols: imap, graph, ews".to_string()),
        ],
        "CTO_JAMI_ACCOUNT_ID" | "CTO_JAMI_PROFILE_NAME" => vec![
            Line::from(String::new()),
            Line::from("mobile  Jami in App Store / Google Play".to_string()),
        ],
        "CTOX_OWNER_PREFERRED_CHANNEL" => vec![
            Line::from(String::new()),
            Line::from("Applies the channel block below this row.".to_string()),
        ],
        _ => Vec::new(),
    }
}

fn signed_delta(delta: i64) -> String {
    if delta > 0 {
        format!("+{delta}")
    } else {
        delta.to_string()
    }
}

fn header_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    vec![
        combined_gpu_bar_line(app, width),
        combined_gpu_label_line(app, width),
        model_mode_line(app, width),
        context_window_line(app, width),
        context_label_line(app, width),
    ]
}

fn model_mode_line(app: &App, width: usize) -> Line<'static> {
    let name_width = width.clamp(16, 32) / 3;
    let active_model = compact_model_name(&app.header.model, name_width);
    let base_model = compact_model_name(&app.header.base_model, name_width);
    let boost_model = app
        .header
        .boost_model
        .as_deref()
        .map(|value| compact_model_name(value, name_width))
        .filter(|value| !value.is_empty());
    let mut text = format!("base {base_model}  active {active_model}");
    if app.header.boost_active {
        if let Some(boost_model) = boost_model {
            text.push_str(&format!("  boost {boost_model}"));
        } else {
            text.push_str("  boost on");
        }
        if let Some(remaining_seconds) = app.header.boost_remaining_seconds {
            let remaining_minutes = (remaining_seconds + 59) / 60;
            text.push_str(&format!("  {}m left", remaining_minutes));
        }
        if let Some(reason) = app.header.boost_reason.as_deref() {
            let room = width.saturating_sub(text.chars().count() + 2);
            if room > 8 {
                text.push_str(&format!("  {}", truncate_line(reason, room)));
            }
        }
    } else if let Some(boost_model) = boost_model {
        text.push_str(&format!("  boost {boost_model} idle"));
    }
    Line::from(truncate_line(&text, width))
}

fn render_settings_view_tabs(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let selected = match app.settings_view {
        SettingsView::General => 0,
        SettingsView::Model => 1,
    };
    let titles = ["General", "Model"]
        .into_iter()
        .map(Line::from)
        .collect::<Vec<_>>();
    let widget = Tabs::new(titles)
        .select(selected)
        .divider(" ")
        .padding("", "")
        .style(Style::default().fg(Color::DarkGray).bg(Color::Rgb(8, 8, 8)))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(widget, area);
}

fn combined_gpu_bar_line(app: &App, width: usize) -> Line<'static> {
    if app.header.gpu_cards.is_empty() {
        return Line::from(truncate_line("GPU telemetry unavailable", width));
    }
    let per_gpu_width = gpu_segment_width(app, width);
    let mut spans = Vec::new();
    for (idx, card) in app.header.gpu_cards.iter().enumerate() {
        if idx > 0 {
            spans.push(Span::raw("  "));
        }
        spans.extend(gpu_usage_bar_spans(card, per_gpu_width));
    }
    truncate_line_spans(spans, width)
}

fn combined_gpu_label_line(app: &App, width: usize) -> Line<'static> {
    if app.header.gpu_cards.is_empty() {
        return Line::from(String::new());
    }
    let per_gpu_width = gpu_segment_width(app, width);
    let mut text = String::new();
    for (idx, card) in app.header.gpu_cards.iter().enumerate() {
        if idx > 0 {
            text.push_str("  ");
        }
        let segment = format!(
            "GPU{} {}/{}G {}%",
            card.index,
            card.used_mb / 1024,
            card.total_mb / 1024,
            card.utilization
        );
        text.push_str(&pad_to_width(&segment, per_gpu_width));
    }
    if let Some(avg_tps) = app.header.avg_tokens_per_second {
        let suffix = format!(" {:>3.0} tok/s", avg_tps);
        if text.chars().count() + suffix.chars().count() <= width {
            text.push_str(&suffix);
        }
    }
    Line::from(truncate_line(&text, width))
}

fn gpu_segment_width(app: &App, width: usize) -> usize {
    let gpu_count = app.header.gpu_cards.len().max(1);
    ((width.saturating_sub(gpu_count.saturating_sub(1) * 2)) / gpu_count).clamp(10, 24)
}

fn pad_to_width(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len >= width {
        truncate_line(text, width)
    } else {
        format!("{text}{:width$}", "", width = width - len)
    }
}

fn context_window_line(app: &App, width: usize) -> Line<'static> {
    let bar_width = width.saturating_sub(2).max(16);
    let compact_percent = value_for_key(app, "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT")
        .parse::<usize>()
        .ok()
        .unwrap_or(75);
    let display_compact = app.header.max_context.saturating_mul(compact_percent) / 100;
    let fill_index = ratio_index(app.header.current_tokens, app.header.max_context, bar_width);
    let compact_index = ratio_index(display_compact, app.header.max_context, bar_width);
    let configured_index = ratio_index(
        app.header.configured_context,
        app.header.max_context,
        bar_width,
    );
    let chat_color = role_color("chat");
    let mut spans = vec![Span::raw("▕")];
    for idx in 0..bar_width {
        let bg = if idx < fill_index {
            chat_color
        } else {
            Color::Rgb(32, 32, 32)
        };
        if idx == compact_index.min(bar_width.saturating_sub(1)) {
            spans.push(Span::styled("◆", Style::default().fg(Color::White).bg(bg)));
        } else if idx == configured_index.min(bar_width.saturating_sub(1)) {
            spans.push(Span::styled("▲", Style::default().fg(chat_color).bg(bg)));
        } else {
            spans.push(Span::styled(" ", Style::default().bg(bg)));
        }
    }
    spans.push(Span::raw("▏"));
    truncate_line_spans(spans, width)
}

fn context_label_line(app: &App, width: usize) -> Line<'static> {
    let compact_percent = value_for_key(app, "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT")
        .parse::<usize>()
        .ok()
        .unwrap_or(75);
    let display_compact = app.header.max_context.saturating_mul(compact_percent) / 100;
    let text = format!(
        "{}k compact   {}k cfg   {}k max",
        display_compact / 1024,
        app.header.configured_context / 1024,
        app.header.max_context / 1024
    );
    Line::from(truncate_line(&text, width))
}

fn gpu_usage_bar_spans(card: &super::GpuCardState, width: usize) -> Vec<Span<'static>> {
    let bar_width = width.saturating_sub(2).clamp(8, 24);
    if card.total_mb == 0 {
        return vec![Span::styled(
            "[no-gpu-data]",
            Style::default().fg(Color::DarkGray),
        )];
    }

    let mut spans = Vec::new();
    spans.push(Span::styled("[", Style::default().fg(Color::Gray)));
    let mut painted = 0usize;
    let total_mb = card.total_mb.max(1);

    for allocation in &card.allocations {
        let seg =
            ((allocation.used_mb as f64 / total_mb as f64) * bar_width as f64).round() as usize;
        let seg = seg.max(1);
        painted = painted.saturating_add(seg);
        spans.push(Span::styled(
            " ".repeat(seg),
            Style::default().bg(model_color(&allocation.model)),
        ));
    }

    if painted < bar_width {
        spans.push(Span::styled(
            " ".repeat(bar_width - painted),
            Style::default().bg(Color::Rgb(32, 32, 32)),
        ));
    }
    spans.push(Span::styled("]", Style::default().fg(Color::Gray)));
    spans
}

fn model_color(model: &str) -> Color {
    role_color(model_role(model))
}

fn model_role(model: &str) -> &'static str {
    let lower = model.to_ascii_lowercase();
    if lower.contains("embedding") {
        "embed"
    } else if lower.contains("voxtral") || lower.contains("stt") {
        "stt"
    } else if lower.contains("tts") {
        "tts"
    } else {
        "chat"
    }
}

fn role_color(role: &str) -> Color {
    match role {
        "embed" => Color::Yellow,
        "stt" => Color::LightMagenta,
        "tts" => Color::LightGreen,
        _ => Color::Cyan,
    }
}

fn setting_row_style(key: &str, _value: &str) -> Style {
    let color = match key {
        "CTOX_CHAT_SOURCE" | "CTOX_API_PROVIDER" | "CTOX_CHAT_MODEL" | "CTOX_CHAT_LOCAL_PRESET" => {
            role_color("chat")
        }
        "CTOX_EMBEDDING_MODEL" => role_color("embed"),
        "CTOX_STT_MODEL" => role_color("stt"),
        "CTOX_TTS_MODEL" => role_color("tts"),
        "CTOX_OWNER_NAME"
        | "CTOX_OWNER_PREFERRED_CHANNEL"
        | "CTO_EMAIL_ADDRESS"
        | "CTO_EMAIL_PROVIDER"
        | "CTO_EMAIL_IMAP_HOST"
        | "CTO_EMAIL_IMAP_PORT"
        | "CTO_EMAIL_SMTP_HOST"
        | "CTO_EMAIL_SMTP_PORT"
        | "CTO_EMAIL_GRAPH_USER"
        | "CTO_EMAIL_EWS_URL"
        | "CTO_EMAIL_EWS_AUTH_TYPE"
        | "CTO_EMAIL_EWS_USERNAME"
        | "CTO_JAMI_PROFILE_NAME"
        | "CTO_JAMI_ACCOUNT_ID" => Color::LightBlue,
        _ => Color::White,
    };
    Style::default().fg(color)
}

fn ratio_index(value: usize, total: usize, width: usize) -> usize {
    if total == 0 || width == 0 {
        0
    } else {
        ((value.min(total) as f64 / total as f64) * width as f64).floor() as usize
    }
}

fn header_preview_line(app: &App, width: usize) -> Line<'static> {
    let model_item = app
        .settings_items
        .iter()
        .find(|item| item.key == "CTOX_CHAT_MODEL");
    let compact_item = app
        .settings_items
        .iter()
        .find(|item| item.key == "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT");
    let channel_item = app
        .settings_items
        .iter()
        .find(|item| item.key == "CTOX_OWNER_PREFERRED_CHANNEL");
    if let (Some(model_item), Some(compact_item)) = (model_item, compact_item) {
        if app.setting_is_dirty(model_item) {
            let loaded = execution_baseline::model_profile_for_model(&model_item.saved_value).ok();
            let draft = execution_baseline::model_profile_for_model(&model_item.value).ok();
            if let (Some(loaded), Some(draft)) = (loaded, draft) {
                let compact_percent = compact_item
                    .value
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .unwrap_or(75);
                let draft_compact = (draft.family_profile.max_seq_len as usize)
                    .saturating_mul(compact_percent)
                    / 100;
                let ctx_delta = draft.family_profile.max_seq_len as i64
                    - loaded.family_profile.max_seq_len as i64;
                let compact_delta = draft_compact as i64 - app.header.compact_at as i64;
                let loaded_tps = app
                    .model_perf_stats
                    .get(model_item.saved_value.trim())
                    .map(|stats| stats.avg_tokens_per_second);
                let draft_tps = app
                    .model_perf_stats
                    .get(model_item.value.trim())
                    .map(|stats| stats.avg_tokens_per_second);
                return truncate_line_spans(
                    vec![
                        Span::styled(
                            "draft ",
                            Style::default()
                                .fg(Color::LightYellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            compact_model_name(&model_item.value, 18),
                            Style::default().fg(Color::LightCyan),
                        ),
                        Span::raw("  ctx "),
                        Span::raw(draft.family_profile.max_seq_len.to_string()),
                        delta_span(ctx_delta),
                        Span::raw("  compact "),
                        Span::raw(draft_compact.to_string()),
                        delta_span(compact_delta),
                        Span::raw("  avg tok/s "),
                        Span::styled(
                            draft_tps
                                .map(|value| format!("{value:.1}"))
                                .unwrap_or_else(|| "-".to_string()),
                            Style::default().fg(Color::White),
                        ),
                        delta_float_span(
                            draft_tps
                                .zip(loaded_tps)
                                .map(|(draft, loaded)| draft - loaded),
                        ),
                    ],
                    width,
                );
            }
        }
        if app.setting_is_dirty(compact_item) {
            let draft_percent = compact_item
                .value
                .trim()
                .parse::<usize>()
                .ok()
                .unwrap_or(75);
            let draft_compact = app.header.realized_context.saturating_mul(draft_percent) / 100;
            return truncate_line_spans(
                vec![
                    Span::styled(
                        "draft ",
                        Style::default()
                            .fg(Color::LightYellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("compact "),
                    Span::raw(format!("{draft_percent}% -> {draft_compact}")),
                    delta_span(draft_compact as i64 - app.header.compact_at as i64),
                ],
                width,
            );
        }
    }
    if let Some(channel_item) = channel_item {
        if app.setting_is_dirty(channel_item) {
            return truncate_line_spans(
                vec![
                    Span::styled(
                        "draft ",
                        Style::default()
                            .fg(Color::LightYellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("channel "),
                    Span::styled(
                        channel_item.value.trim().to_string(),
                        Style::default().fg(Color::LightCyan),
                    ),
                ],
                width,
            );
        }
    }
    Line::from(vec![Span::styled(
        "loaded state",
        Style::default().fg(Color::DarkGray),
    )])
}

fn delta_span(delta: i64) -> Span<'static> {
    let style = if delta > 0 {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else if delta < 0 {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Span::styled(format!(" {}", signed_delta(delta)), style)
}

fn delta_float_span(delta: Option<f64>) -> Span<'static> {
    match delta {
        Some(value) if value > 0.0 => Span::styled(
            format!(" +{value:.1}"),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Some(value) if value < 0.0 => Span::styled(
            format!(" {value:.1}"),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Some(_) => Span::styled(" 0.0".to_string(), Style::default().fg(Color::DarkGray)),
        None => Span::styled(" -".to_string(), Style::default().fg(Color::DarkGray)),
    }
}

fn truncate_line_spans(spans: Vec<Span<'static>>, max_chars: usize) -> Line<'static> {
    if max_chars == 0 {
        return Line::from(String::new());
    }
    let text = spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    if text.chars().count() <= max_chars {
        return Line::from(spans);
    }
    let truncated = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>()
        + "…";
    Line::from(truncated)
}

fn display_setting_value(value: &str, secret: bool) -> String {
    if value.is_empty() {
        "-".to_string()
    } else if secret {
        "********".to_string()
    } else {
        value.to_string()
    }
}

fn truncate_line(value: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let mut out = collapsed
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_app() -> App {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut db_path = std::env::temp_dir();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        db_path.push(format!("ctox-tui-render-{stamp}.db"));
        let mut app = App::new(root, db_path);
        app.page = Page::Chat;
        app
    }

    fn buffer_text(buffer: &ratatui::buffer::Buffer) -> String {
        let area = buffer.area;
        let mut out = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn chat_sidebar_footer_visible_in_24_rows() {
        let backend = TestBackend::new(150, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = test_app();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("interrupt"), "{text}");
        assert!(text.contains("next page"), "{text}");
        assert!(text.contains("quit"), "{text}");
    }

    #[test]
    fn chat_view_shows_turn_state_and_structured_roles() {
        let backend = TestBackend::new(140, 36);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = test_app();
        app.service_status.running = true;
        app.service_status.busy = true;
        app.service_status.active_source_label = Some("queue".to_string());
        app.service_status.pending_count = 2;
        app.service_status.last_reply_chars = Some(534);
        app.service_status.recent_events = vec![
            "Started queued queue prompt".to_string(),
            "Completed queue reply with 318 chars".to_string(),
        ];
        app.chat_messages.push(crate::lcm::MessageRecord {
            message_id: 1,
            conversation_id: 1,
            seq: 1,
            role: "user".to_string(),
            content: "Installiere Nextcloud.".to_string(),
            created_at: "2026-03-26T10:00:00Z".to_string(),
            token_count: 10,
        });
        app.chat_messages.push(crate::lcm::MessageRecord {
            message_id: 2,
            conversation_id: 1,
            seq: 2,
            role: "assistant".to_string(),
            content: "Nextcloud bleibt blockiert.".to_string(),
            created_at: "2026-03-26T10:00:01Z".to_string(),
            token_count: 12,
        });
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("turn"), "{text}");
        assert!(text.contains("working"), "{text}");
        assert!(text.contains("queue"), "{text}");
        assert!(text.contains("CTOX"), "{text}");
        assert!(text.contains("YOU"), "{text}");
    }

    #[test]
    fn chat_header_shows_boost_state() {
        let backend = TestBackend::new(140, 18);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = test_app();
        app.header.model = "gpt-5.4".to_string();
        app.header.base_model = "gpt-5.4-mini".to_string();
        app.header.boost_model = Some("gpt-5.4".to_string());
        app.header.boost_active = true;
        app.header.boost_remaining_seconds = Some(7 * 60);
        app.header.boost_reason = Some("stuck in repair loop".to_string());
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("base gpt-5.4-mini"), "{text}");
        assert!(text.contains("boost gpt-5.4"), "{text}");
        assert!(text.contains("7m left"), "{text}");
    }

    #[test]
    fn skills_view_shows_skill_details_and_resources() {
        let backend = TestBackend::new(140, 36);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = test_app();
        app.page = Page::Skills;
        app.skill_catalog = vec![super::super::SkillCatalogEntry {
            name: "service-deployment".to_string(),
            source: "ctox system".to_string(),
            skill_path: PathBuf::from("/tmp/service-deployment/SKILL.md"),
            description: "Use when CTOX needs to install, configure, start and verify software."
                .to_string(),
            helper_tools: vec!["deployment_bootstrap.py".to_string()],
            resources: vec![
                "references: deployment-rules.md, install-patterns.md".to_string(),
                "agents: openai.yaml".to_string(),
            ],
        }];
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("Skills"), "{text}");
        assert!(text.contains("service-deployment"), "{text}");
        assert!(text.contains("deployment_bootstrap.py"), "{text}");
        assert!(text.contains("install-patterns.md"), "{text}");
    }
}
