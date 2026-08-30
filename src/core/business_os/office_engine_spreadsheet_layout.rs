// Origin: CTOX
// License: AGPL-3.0-only

use super::*;

pub(super) fn replace_spreadsheet_print_layout(
    xml: Vec<u8>,
    print: &DecodedSpreadsheetPrintPivot,
) -> anyhow::Result<Vec<u8>> {
    let mut output = String::from_utf8(xml).context("worksheet print layout is not UTF-8")?;
    output = replace_or_insert_worksheet_node(
        output,
        "sheetPr",
        &spreadsheet_sheet_pr_xml(print),
        "<dimension",
    )?;
    output = replace_or_insert_worksheet_node(
        output,
        "sheetViews",
        &spreadsheet_sheet_views_xml(print),
        "<sheetFormatPr",
    )?;
    output = replace_or_insert_worksheet_node(
        output,
        "printOptions",
        &spreadsheet_print_options_xml(print),
        "<pageMargins",
    )?;
    output = replace_or_insert_worksheet_node(
        output,
        "pageMargins",
        &spreadsheet_page_margins_xml(print),
        "<pageSetup",
    )?;
    output = replace_or_insert_worksheet_node(
        output,
        "pageSetup",
        &spreadsheet_page_setup_xml(print),
        "<headerFooter",
    )?;
    output = replace_or_insert_worksheet_node(
        output,
        "headerFooter",
        &spreadsheet_header_footer_xml(print),
        "<rowBreaks",
    )?;
    output = replace_or_insert_worksheet_node(
        output,
        "rowBreaks",
        &spreadsheet_breaks_xml("rowBreaks", print.row_breaks.as_ref()),
        "<colBreaks",
    )?;
    output = replace_or_insert_worksheet_node(
        output,
        "colBreaks",
        &spreadsheet_breaks_xml("colBreaks", print.col_breaks.as_ref()),
        "<pivotTableParts",
    )?;
    Ok(output.into_bytes())
}

fn replace_or_insert_worksheet_node(
    mut xml: String,
    name: &str,
    replacement: &str,
    before: &str,
) -> anyhow::Result<String> {
    let regex = Regex::new(&format!(r#"(?s)<{name}\b[^>]*(?:/>|>.*?</{name}>)"#))?;
    if regex.is_match(&xml) {
        return Ok(regex.replace(&xml, replacement).into_owned());
    }
    if replacement.is_empty() {
        return Ok(xml);
    }
    if let Some(position) = xml.find(before) {
        xml.insert_str(position, replacement);
        return Ok(xml);
    }
    let position = xml
        .rfind("</worksheet>")
        .context("worksheet closing element missing")?;
    xml.insert_str(position, replacement);
    Ok(xml)
}

fn spreadsheet_sheet_pr_xml(print: &DecodedSpreadsheetPrintPivot) -> String {
    print
        .fit_to_page
        .map(|value| {
            format!(
                r#"<sheetPr><pageSetUpPr fitToPage="{}"/></sheetPr>"#,
                u8::from(value)
            )
        })
        .unwrap_or_default()
}

fn spreadsheet_sheet_views_xml(print: &DecodedSpreadsheetPrintPivot) -> String {
    print
        .view
        .map(|view| {
            format!(
                r#"<sheetViews><sheetView workbookViewId="0" view="{}"/></sheetViews>"#,
                ["normal", "pageBreakPreview", "pageLayout"]
                    .get(view as usize)
                    .unwrap_or(&"normal")
            )
        })
        .unwrap_or_default()
}

fn spreadsheet_print_options_xml(print: &DecodedSpreadsheetPrintPivot) -> String {
    let names = [
        "gridLines",
        "headings",
        "gridLinesSet",
        "horizontalCentered",
        "verticalCentered",
    ];
    let attrs = print
        .print_options
        .iter()
        .zip(names)
        .filter_map(|(value, name)| value.map(|value| format!(r#" {name}="{}""#, u8::from(value))))
        .collect::<String>();
    if attrs.is_empty() {
        String::new()
    } else {
        format!("<printOptions{attrs}/>")
    }
}

fn spreadsheet_page_margins_xml(print: &DecodedSpreadsheetPrintPivot) -> String {
    let names = ["left", "top", "right", "bottom", "header", "footer"];
    let attrs = print
        .margins
        .iter()
        .zip(names)
        .filter_map(|(value, name)| value.map(|value| format!(r#" {name}="{value}""#)))
        .collect::<String>();
    if attrs.is_empty() {
        String::new()
    } else {
        format!("<pageMargins{attrs}/>")
    }
}

fn spreadsheet_page_setup_xml(print: &DecodedSpreadsheetPrintPivot) -> String {
    let mut attrs = String::new();
    for (property, value) in &print.page_setup {
        let (name, text) = match (*property, value.as_slice()) {
            (0, [value]) => (
                "orientation",
                if *value == 0 {
                    "landscape".to_string()
                } else {
                    "portrait".to_string()
                },
            ),
            (1, [value]) => ("paperSize", value.to_string()),
            (2, [value]) => ("blackAndWhite", u8::from(*value != 0).to_string()),
            (3, [value]) => ("cellComments", value.to_string()),
            (4 | 7 | 8 | 9 | 10 | 15 | 18, bytes) if bytes.len() == 4 => (
                {
                    let name = match property {
                        4 => "copies",
                        7 => "firstPageNumber",
                        8 => "fitToHeight",
                        9 => "fitToWidth",
                        10 => "horizontalDpi",
                        15 => "scale",
                        _ => "verticalDpi",
                    };
                    name
                },
                u32::from_le_bytes(bytes.try_into().unwrap()).to_string(),
            ),
            (5, [value]) => ("draft", u8::from(*value != 0).to_string()),
            (6, [value]) => ("errors", value.to_string()),
            (11, [value]) => (
                "pageOrder",
                if *value == 0 {
                    "downThenOver".to_string()
                } else {
                    "overThenDown".to_string()
                },
            ),
            (16, [value]) => ("useFirstPageNumber", u8::from(*value != 0).to_string()),
            (17, [value]) => ("usePrinterDefaults", u8::from(*value != 0).to_string()),
            _ => continue,
        };
        attrs.push_str(&format!(r#" {name}="{text}""#));
    }
    if attrs.is_empty() {
        String::new()
    } else {
        format!("<pageSetup{attrs}/>")
    }
}

fn spreadsheet_header_footer_xml(print: &DecodedSpreadsheetPrintPivot) -> String {
    if print.header_footer_flags.is_empty() && print.header_footer_text.is_empty() {
        return String::new();
    }
    let flag_names = [
        "alignWithMargins",
        "differentFirst",
        "differentOddEven",
        "scaleWithDoc",
    ];
    let text_names = [
        "",
        "",
        "",
        "",
        "evenFooter",
        "evenHeader",
        "firstFooter",
        "firstHeader",
        "oddFooter",
        "oddHeader",
    ];
    let flags = print
        .header_footer_flags
        .iter()
        .filter_map(|(property, value)| {
            flag_names
                .get(*property as usize)
                .map(|name| format!(r#" {name}="{}""#, u8::from(*value)))
        })
        .collect::<String>();
    let children = print
        .header_footer_text
        .iter()
        .filter_map(|(property, value)| {
            text_names
                .get(*property as usize)
                .filter(|name| !name.is_empty())
                .map(|name| format!("<{name}>{}</{name}>", xml_escape_text(value)))
        })
        .collect::<String>();
    format!("<headerFooter{flags}>{children}</headerFooter>")
}

fn spreadsheet_breaks_xml(name: &str, breaks: Option<&SpreadsheetBreaks>) -> String {
    let Some(breaks) = breaks else {
        return String::new();
    };
    let children = breaks
        .breaks
        .iter()
        .map(|item| {
            format!(
                r#"<brk id="{}" min="{}" max="{}" man="{}"/>"#,
                item.id,
                item.min,
                item.max,
                u8::from(item.manual)
            )
        })
        .collect::<String>();
    format!(
        r#"<{name} count="{}" manualBreakCount="{}">{children}</{name}>"#,
        breaks.count, breaks.manual_count
    )
}

pub(super) fn replace_spreadsheet_drawing_anchor(
    xml: Vec<u8>,
    drawing: &DecodedSpreadsheetChartDrawing,
) -> anyhow::Result<Vec<u8>> {
    let mut output = String::from_utf8(xml).context("spreadsheet drawing is not UTF-8")?;
    let from = spreadsheet_drawing_point_xml(
        "from",
        drawing.from_col,
        drawing.from_col_off_mm,
        drawing.from_row,
        drawing.from_row_off_mm,
    );
    let to = spreadsheet_drawing_point_xml(
        "to",
        drawing.to_col,
        drawing.to_col_off_mm,
        drawing.to_row,
        drawing.to_row_off_mm,
    );
    let from_regex = Regex::new(r#"(?s)<xdr:from>.*?</xdr:from>"#)?;
    let to_regex = Regex::new(r#"(?s)<xdr:to>.*?</xdr:to>"#)?;
    ensure!(
        from_regex.is_match(&output),
        "drawing has no xdr:from anchor"
    );
    ensure!(to_regex.is_match(&output), "drawing has no xdr:to anchor");
    output = from_regex.replace(&output, from).into_owned();
    output = to_regex.replace(&output, to).into_owned();
    if let Some([off_x, off_y, ext_x, ext_y]) = drawing.xfrm_emu {
        let off_regex = Regex::new(r#"<a:off\b[^>]*/>"#)?;
        let ext_regex = Regex::new(r#"<a:ext\b[^>]*/>"#)?;
        ensure!(
            off_regex.is_match(&output),
            "drawing has no a:off transform"
        );
        ensure!(
            ext_regex.is_match(&output),
            "drawing has no a:ext transform"
        );
        output = off_regex
            .replace(&output, format!(r#"<a:off x="{off_x}" y="{off_y}"/>"#))
            .into_owned();
        output = ext_regex
            .replace(&output, format!(r#"<a:ext cx="{ext_x}" cy="{ext_y}"/>"#))
            .into_owned();
    }
    Ok(output.into_bytes())
}

fn spreadsheet_drawing_point_xml(
    name: &str,
    column: u32,
    column_offset_mm: f64,
    row: u32,
    row_offset_mm: f64,
) -> String {
    let to_emu = |value: f64| (value * 36_000.0).round() as i64;
    format!(
        "<xdr:{name}><xdr:col>{column}</xdr:col><xdr:colOff>{}</xdr:colOff><xdr:row>{row}</xdr:row><xdr:rowOff>{}</xdr:rowOff></xdr:{name}>",
        to_emu(column_offset_mm),
        to_emu(row_offset_mm),
    )
}

pub(super) fn replace_spreadsheet_chart_style(
    xml: Vec<u8>,
    style: Option<u8>,
) -> anyhow::Result<Vec<u8>> {
    let mut output = String::from_utf8(xml).context("spreadsheet chart is not UTF-8")?;
    let style_regex = Regex::new(r#"<c:style\b[^>]*/>"#)?;
    output = style_regex.replace_all(&output, "").into_owned();
    if let Some(style) = style {
        let position = output
            .find("<c:chart>")
            .context("chart space has no c:chart element")?;
        output.insert_str(position, &format!("<c:style val=\"{style}\"/>"));
    }
    Ok(output.into_bytes())
}

pub(super) fn replace_spreadsheet_workbook_metadata(
    xml: Vec<u8>,
    names: &[EditorDefinedNameManifest],
    protection: Option<&EditorWorkbookProtectionManifest>,
) -> anyhow::Result<Vec<u8>> {
    let mut output = String::from_utf8(xml).context("workbook is not UTF-8")?;
    let names_regex = Regex::new(r#"(?s)<definedNames\b[^>]*>.*?</definedNames>"#)?;
    output = names_regex.replace_all(&output, "").into_owned();
    let protection_regex = Regex::new(r#"<workbookProtection\b[^>]*/>"#)?;
    output = protection_regex.replace_all(&output, "").into_owned();
    let mut replacement = String::new();
    if let Some(value) = protection {
        replacement.push_str("<workbookProtection");
        if let Some(password) = value.password.as_deref() {
            replacement.push_str(&format!(
                " workbookPassword=\"{}\"",
                escape_xml_attribute(password)
            ));
        }
        for (name, enabled) in [
            ("lockStructure", value.lock_structure),
            ("lockWindows", value.lock_windows),
            ("lockRevision", value.lock_revision),
        ] {
            if enabled {
                replacement.push_str(&format!(" {name}=\"1\""));
            }
        }
        replacement.push_str("/>");
    }
    if !names.is_empty() {
        replacement.push_str("<definedNames>");
        for name in names {
            replacement.push_str(&format!(
                "<definedName name=\"{}\"",
                escape_xml_attribute(&name.name)
            ));
            if let Some(local) = name.local_sheet_id {
                replacement.push_str(&format!(" localSheetId=\"{local}\""));
            }
            if name.hidden {
                replacement.push_str(" hidden=\"1\"");
            }
            replacement.push_str(&format!(
                ">{}</definedName>",
                escape_xml_text(&name.reference)
            ));
        }
        replacement.push_str("</definedNames>");
    }
    let position = output
        .find("</workbook>")
        .context("workbook closing element is missing")?;
    output.insert_str(position, &replacement);
    Ok(output.into_bytes())
}

pub(super) fn replace_spreadsheet_sheet_protection(
    xml: Vec<u8>,
    protection: Option<&EditorSheetProtectionManifest>,
) -> anyhow::Result<Vec<u8>> {
    let mut output = String::from_utf8(xml).context("worksheet is not UTF-8")?;
    let regex = Regex::new(r#"<sheetProtection\b[^>]*/>"#)?;
    let Some(value) = protection else {
        return Ok(regex.replace_all(&output, "").into_owned().into_bytes());
    };
    let mut element = String::from("<sheetProtection");
    if let Some(password) = value.password.as_deref() {
        element.push_str(&format!(" password=\"{}\"", escape_xml_attribute(password)));
    }
    for (name, enabled) in [
        ("sheet", value.sheet),
        ("objects", value.objects),
        ("scenarios", value.scenarios),
        ("formatCells", value.format_cells),
        ("formatColumns", value.format_columns),
        ("formatRows", value.format_rows),
        ("insertColumns", value.insert_columns),
        ("insertHyperlinks", value.insert_hyperlinks),
        ("insertRows", value.insert_rows),
        ("deleteColumns", value.delete_columns),
        ("deleteRows", value.delete_rows),
        ("selectLockedCells", value.select_locked_cells),
        ("sort", value.sort),
        ("autoFilter", value.auto_filter),
        ("pivotTables", value.pivot_tables),
        ("selectUnlockedCells", value.select_unlocked_cells),
    ] {
        element.push_str(&format!(" {name}=\"{}\"", u8::from(enabled)));
    }
    element.push_str("/>");
    if regex.is_match(&output) {
        return Ok(regex
            .replace_all(&output, element.as_str())
            .into_owned()
            .into_bytes());
    }
    let position = output
        .find("</worksheet>")
        .context("worksheet closing element is missing")?;
    output.insert_str(position, &element);
    Ok(output.into_bytes())
}

pub(super) fn spreadsheet_comment_path<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    worksheet_path: &str,
) -> anyhow::Result<Option<String>> {
    spreadsheet_worksheet_relationship_path(archive, worksheet_path, "/comments")
}

pub(super) fn spreadsheet_vml_path<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    worksheet_path: &str,
) -> anyhow::Result<Option<String>> {
    spreadsheet_worksheet_relationship_path(archive, worksheet_path, "/vmlDrawing")
}

pub(super) fn spreadsheet_worksheet_relationship_path<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    worksheet_path: &str,
    relationship_suffix: &str,
) -> anyhow::Result<Option<String>> {
    let worksheet = Path::new(worksheet_path);
    let filename = worksheet
        .file_name()
        .and_then(|value| value.to_str())
        .context("worksheet path has no filename")?;
    let parent = worksheet
        .parent()
        .and_then(|value| value.to_str())
        .context("worksheet path has no parent")?;
    let Some(xml) = read_optional_zip_part(archive, &format!("{parent}/_rels/{filename}.rels"))?
    else {
        return Ok(None);
    };
    let document = roxmltree::Document::parse(
        std::str::from_utf8(&xml).context("worksheet relationships are not UTF-8")?,
    )?;
    document
        .descendants()
        .find(|node| {
            node.is_element()
                && node.tag_name().name() == "Relationship"
                && node
                    .attribute("Type")
                    .is_some_and(|value| value.ends_with(relationship_suffix))
        })
        .and_then(|node| node.attribute("Target"))
        .map(|target| normalize_ooxml_relationship_target(parent, target))
        .transpose()
}

pub(super) fn write_ooxml_spreadsheet_comment_vml(
    comments: &[EditorSpreadsheetCommentManifest],
) -> anyhow::Result<Vec<u8>> {
    let mut output = String::from(
        r#"<xml xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel"><o:shapelayout v:ext="edit"><o:idmap v:ext="edit" data="1"/></o:shapelayout><v:shapetype id="_x0000_t202" coordsize="21600,21600" o:spt="202" path="m,l,21600r21600,l21600,xe"><v:stroke joinstyle="miter"/><v:path gradientshapeok="t" o:connecttype="rect"/></v:shapetype>"#,
    );
    for (index, comment) in comments.iter().enumerate() {
        let column = parse_cell_column(&comment.reference)?;
        let row = comment
            .reference
            .chars()
            .skip_while(|value| value.is_ascii_alphabetic())
            .collect::<String>()
            .parse::<u32>()
            .context("spreadsheet comment row is invalid")?
            .checked_sub(1)
            .context("spreadsheet comment row must be positive")?;
        output.push_str(&format!(
            r##"<v:shape id="_x0000_s{}" type="#_x0000_t202" style="position:absolute;margin-left:80pt;margin-top:5pt;width:108pt;height:59pt;z-index:{};visibility:hidden" fillcolor="#ffffe1" o:insetmode="auto"><v:fill color2="#ffffe1"/><v:shadow on="t" color="black" obscured="t"/><v:path o:connecttype="none"/><v:textbox style="mso-direction-alt:auto"><div style="text-align:left"/></v:textbox><x:ClientData ObjectType="Note"><x:MoveWithCells/><x:SizeWithCells/><x:Anchor>{}, 15, {}, 2, {}, 31, {}, 1</x:Anchor><x:AutoFill>False</x:AutoFill><x:Row>{}</x:Row><x:Column>{}</x:Column></x:ClientData></v:shape>"##,
            1025 + index,
            index + 1,
            column,
            row,
            column + 2,
            row + 3,
            row,
            column,
        ));
    }
    output.push_str("</xml>");
    Ok(output.into_bytes())
}

pub(super) fn write_ooxml_spreadsheet_comments(
    comments: &[EditorSpreadsheetCommentManifest],
) -> Vec<u8> {
    let mut authors = Vec::<String>::new();
    for comment in comments {
        if !authors.contains(&comment.author) {
            authors.push(comment.author.clone());
        }
    }
    let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><comments xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><authors>");
    for author in &authors {
        output.push_str(&format!("<author>{}</author>", escape_xml_text(author)));
    }
    output.push_str("</authors><commentList>");
    for comment in comments {
        let author_id = authors
            .iter()
            .position(|author| author == &comment.author)
            .unwrap_or(0);
        output.push_str(&format!(
            "<comment ref=\"{}\" authorId=\"{author_id}\"><text><t>{}</t></text></comment>",
            escape_xml_attribute(&comment.reference),
            escape_xml_text(&comment.text)
        ));
    }
    output.push_str("</commentList></comments>");
    output.into_bytes()
}

pub(super) fn replace_spreadsheet_validation_conditional(
    xml: Vec<u8>,
    before: &EditorWorksheetManifest,
    after: &EditorWorksheetManifest,
    differential_formats: &[EditorDifferentialStyleManifest],
) -> anyhow::Result<Vec<u8>> {
    if before.data_validations == after.data_validations
        && before.conditional_formats == after.conditional_formats
    {
        return Ok(xml);
    }
    let mut output = String::from_utf8(xml).context("worksheet is not UTF-8")?;
    let conditional_regex =
        Regex::new(r#"(?s)<conditionalFormatting\b[^>]*>.*?</conditionalFormatting>"#)?;
    output = conditional_regex.replace_all(&output, "").into_owned();
    let validation_regex = Regex::new(r#"(?s)<dataValidations\b[^>]*>.*?</dataValidations>"#)?;
    output = validation_regex.replace_all(&output, "").into_owned();
    let mut replacement = String::new();
    for conditional in &after.conditional_formats {
        replacement.push_str(&write_ooxml_conditional_format(
            conditional,
            differential_formats,
        )?);
    }
    if !after.data_validations.is_empty() {
        replacement.push_str(&format!(
            "<dataValidations count=\"{}\">{}</dataValidations>",
            after.data_validations.len(),
            after
                .data_validations
                .iter()
                .map(write_ooxml_data_validation)
                .collect::<String>()
        ));
    }
    let position = output
        .find("</worksheet>")
        .context("worksheet closing element is missing")?;
    output.insert_str(position, &replacement);
    Ok(output.into_bytes())
}

fn write_ooxml_data_validation(validation: &EditorDataValidationManifest) -> String {
    let mut attributes = vec![
        format!(
            "type=\"{}\"",
            escape_xml_attribute(&validation.validation_type)
        ),
        format!("allowBlank=\"{}\"", u8::from(validation.allow_blank)),
        format!(
            "showErrorMessage=\"{}\"",
            u8::from(validation.show_error_message)
        ),
        format!("sqref=\"{}\"", escape_xml_attribute(&validation.reference)),
    ];
    for (name, value) in [
        ("operator", validation.operator.as_deref()),
        ("errorStyle", validation.error_style.as_deref()),
        ("errorTitle", validation.error_title.as_deref()),
        ("error", validation.error.as_deref()),
    ] {
        if let Some(value) = value {
            attributes.push(format!("{name}=\"{}\"", escape_xml_attribute(value)));
        }
    }
    let formulas = [
        ("formula1", validation.formula1.as_deref()),
        ("formula2", validation.formula2.as_deref()),
    ]
    .into_iter()
    .filter_map(|(name, value)| {
        value.map(|value| format!("<{name}>{}</{name}>", escape_xml_text(value)))
    })
    .collect::<String>();
    format!(
        "<dataValidation {}>{formulas}</dataValidation>",
        attributes.join(" ")
    )
}

fn write_ooxml_conditional_format(
    conditional: &EditorConditionalFormatManifest,
    differential_formats: &[EditorDifferentialStyleManifest],
) -> anyhow::Result<String> {
    let mut attributes = vec![
        format!("type=\"{}\"", escape_xml_attribute(&conditional.rule_type)),
        format!("priority=\"{}\"", conditional.priority),
    ];
    if let Some(operator) = &conditional.operator {
        attributes.push(format!("operator=\"{}\"", escape_xml_attribute(operator)));
    }
    if let Some(style) = &conditional.differential_style {
        let dxf_id = differential_formats
            .iter()
            .position(|candidate| differential_styles_rgb_equal(candidate, style))
            .context("conditional differential style changed but XLSX dxf materialization is not implemented")?;
        attributes.push(format!("dxfId=\"{dxf_id}\""));
    }
    let mut body = conditional
        .formulas
        .iter()
        .map(|formula| format!("<formula>{}</formula>", escape_xml_text(formula)))
        .collect::<String>();
    if conditional.rule_type == "colorScale" {
        body.push_str("<colorScale>");
        for threshold in &conditional.thresholds {
            body.push_str(&format!(
                "<cfvo type=\"{}\"{} />",
                escape_xml_attribute(&threshold.threshold_type),
                threshold
                    .value
                    .as_ref()
                    .map_or_else(String::new, |value| format!(
                        " val=\"{}\"",
                        escape_xml_attribute(value)
                    ))
            ));
        }
        for color in &conditional.colors {
            body.push_str(&format!(
                "<color rgb=\"{}\"/>",
                escape_xml_attribute(&ooxml_argb(color))
            ));
        }
        body.push_str("</colorScale>");
    }
    Ok(format!(
        "<conditionalFormatting sqref=\"{}\"><cfRule {}>{body}</cfRule></conditionalFormatting>",
        escape_xml_attribute(&conditional.reference),
        attributes.join(" ")
    ))
}

fn differential_styles_rgb_equal(
    left: &EditorDifferentialStyleManifest,
    right: &EditorDifferentialStyleManifest,
) -> bool {
    normalized_rgb(&left.fill_rgb) == normalized_rgb(&right.fill_rgb)
        && normalized_rgb(&left.font_rgb) == normalized_rgb(&right.font_rgb)
}

fn normalized_rgb(value: &Option<String>) -> Option<String> {
    value.as_deref().map(|value| {
        value
            .get(value.len().saturating_sub(6)..)
            .unwrap_or(value)
            .to_string()
    })
}

fn ooxml_argb(value: &str) -> String {
    let rgb = value.get(value.len().saturating_sub(6)..).unwrap_or(value);
    format!("FF{rgb}")
}

pub(super) fn spreadsheet_table_paths<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    worksheet_paths: &BTreeMap<String, String>,
) -> anyhow::Result<BTreeMap<String, Vec<String>>> {
    let mut result = BTreeMap::new();
    for (sheet_name, worksheet_path) in worksheet_paths {
        let worksheet = Path::new(worksheet_path);
        let filename = worksheet
            .file_name()
            .and_then(|value| value.to_str())
            .context("worksheet path has no filename")?;
        let parent = worksheet
            .parent()
            .and_then(|value| value.to_str())
            .context("worksheet path has no parent")?;
        let relationships_path = format!("{parent}/_rels/{filename}.rels");
        let Some(relationships) = read_optional_zip_part(archive, &relationships_path)? else {
            result.insert(sheet_name.clone(), Vec::new());
            continue;
        };
        let document = roxmltree::Document::parse(
            std::str::from_utf8(&relationships).context("worksheet relationships are not UTF-8")?,
        )?;
        let paths = document
            .descendants()
            .filter(|node| {
                node.is_element()
                    && node.tag_name().name() == "Relationship"
                    && node
                        .attribute("Type")
                        .is_some_and(|value| value.ends_with("/table"))
            })
            .map(|node| {
                normalize_ooxml_relationship_target(
                    parent,
                    node.attribute("Target")
                        .context("table relationship has no target")?,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        result.insert(sheet_name.clone(), paths);
    }
    Ok(result)
}

pub(super) fn replace_spreadsheet_table_filter_sort(
    xml: &[u8],
    before: &EditorTableManifest,
    after: &EditorTableManifest,
) -> anyhow::Result<Vec<u8>> {
    if before.filters == after.filters && before.sort == after.sort {
        return Ok(xml.to_vec());
    }
    let mut output = std::str::from_utf8(xml)
        .context("spreadsheet table is not UTF-8")?
        .to_string();
    let filters = after
        .filters
        .iter()
        .map(|column| {
            format!(
                "<filterColumn colId=\"{}\"><filters>{}</filters></filterColumn>",
                column.column_id,
                column
                    .values
                    .iter()
                    .map(|value| format!("<filter val=\"{}\"/>", escape_xml_attribute(value)))
                    .collect::<String>()
            )
        })
        .collect::<String>();
    let auto_filter = if filters.is_empty() {
        format!(
            "<autoFilter ref=\"{}\"/>",
            escape_xml_attribute(&after.reference)
        )
    } else {
        format!(
            "<autoFilter ref=\"{}\">{filters}</autoFilter>",
            escape_xml_attribute(&after.reference)
        )
    };
    let auto_filter_regex = Regex::new(r#"(?s)<autoFilter\b[^>]*(?:/>|>.*?</autoFilter>)"#)?;
    let found = auto_filter_regex
        .find(&output)
        .context("table autoFilter element is missing")?;
    output.replace_range(found.range(), &auto_filter);

    let sort_regex = Regex::new(r#"(?s)<sortState\b[^>]*>.*?</sortState>|<sortState\b[^>]*/>"#)?;
    if let Some(found) = sort_regex.find(&output) {
        output.replace_range(found.range(), "");
    }
    if let Some(sort) = &after.sort {
        let sort_xml = format!(
            "<sortState ref=\"{}\"><sortCondition{} ref=\"{}\"/></sortState>",
            escape_xml_attribute(&sort.reference),
            if sort.descending {
                " descending=\"1\""
            } else {
                ""
            },
            escape_xml_attribute(&sort.condition_reference)
        );
        let position = output
            .find("<tableColumns")
            .context("tableColumns element is missing")?;
        output.insert_str(position, &sort_xml);
    }
    Ok(output.into_bytes())
}

pub(super) fn spreadsheet_styles_equivalent(
    before_id: u32,
    before: &SpreadsheetSourceStyles,
    after_id: u32,
    after: &SpreadsheetSourceStyles,
) -> anyhow::Result<bool> {
    let before_xf = before
        .cell_xfs
        .get(before_id as usize)
        .with_context(|| format!("original XLSY style is missing: {before_id}"))?;
    let after_xf = after
        .cell_xfs
        .get(after_id as usize)
        .with_context(|| format!("changed XLSY style is missing: {after_id}"))?;
    let before_font = before
        .fonts
        .get(before_xf.font_id as usize)
        .with_context(|| format!("original XLSY font is missing: {}", before_xf.font_id))?;
    let after_font = after
        .fonts
        .get(after_xf.font_id as usize)
        .with_context(|| format!("changed XLSY font is missing: {}", after_xf.font_id))?;
    let before_num = before.number_formats.get(&before_xf.num_fmt_id);
    let after_num = after.number_formats.get(&after_xf.num_fmt_id);
    let number_format_equal =
        if before_xf.num_fmt_id == after_xf.num_fmt_id && before_xf.num_fmt_id < 164 {
            true
        } else {
            before_num == after_num
        };
    Ok(spreadsheet_fonts_equivalent(before_font, after_font)
        && number_format_equal
        && before_xf.fill_id == after_xf.fill_id
        && before_xf.border_id == after_xf.border_id
        && before_xf.horizontal_alignment == after_xf.horizontal_alignment)
}

pub(super) fn materialize_spreadsheet_styles(
    xml: &[u8],
    changed: &SpreadsheetSourceStyles,
    needed: &BTreeSet<u32>,
) -> anyhow::Result<(Vec<u8>, BTreeMap<u32, u32>)> {
    let mut source = std::str::from_utf8(xml)
        .context("spreadsheet styles are not UTF-8")?
        .to_string();
    let mut materialized = parse_ooxml_styles(xml)?;
    let mut style_map = BTreeMap::new();
    let mut font_entries = Vec::new();
    let mut format_entries = Vec::new();
    let mut xf_entries = Vec::new();
    for style_id in needed {
        let changed_xf = changed
            .cell_xfs
            .get(*style_id as usize)
            .with_context(|| format!("changed XLSY style is missing: {style_id}"))?
            .clone();
        let changed_font = changed
            .fonts
            .get(changed_xf.font_id as usize)
            .with_context(|| format!("changed XLSY font is missing: {}", changed_xf.font_id))?
            .clone();
        let font_id = if let Some(index) = materialized
            .fonts
            .iter()
            .position(|font| spreadsheet_fonts_equivalent(font, &changed_font))
        {
            index as u32
        } else {
            let index = materialized.fonts.len() as u32;
            font_entries.push(spreadsheet_font_xml(&changed_font));
            materialized.fonts.push(changed_font);
            index
        };
        let num_fmt_id = if let Some(code) = changed.number_formats.get(&changed_xf.num_fmt_id) {
            if let Some((id, _)) = materialized
                .number_formats
                .iter()
                .find(|(_, existing)| *existing == code)
            {
                *id
            } else {
                materialized
                    .number_formats
                    .insert(changed_xf.num_fmt_id, code.clone());
                format_entries.push(format!(
                    "<numFmt numFmtId=\"{}\" formatCode=\"{}\"/>",
                    changed_xf.num_fmt_id,
                    escape_xml_attribute(code)
                ));
                changed_xf.num_fmt_id
            }
        } else {
            changed_xf.num_fmt_id
        };
        let mut mapped = changed_xf;
        mapped.font_id = font_id;
        mapped.num_fmt_id = num_fmt_id;
        if let Some(index) = materialized.cell_xfs.iter().position(|xf| xf == &mapped) {
            style_map.insert(*style_id, index as u32);
        } else {
            let index = materialized.cell_xfs.len() as u32;
            xf_entries.push(spreadsheet_xf_xml(&mapped));
            materialized.cell_xfs.push(mapped);
            style_map.insert(*style_id, index);
        }
    }
    if !format_entries.is_empty() {
        if source.contains("</numFmts>") {
            source = append_spreadsheet_style_entries(&source, "numFmts", &format_entries)?;
        } else {
            let root_start = source
                .find("<styleSheet")
                .context("spreadsheet style root is missing")?;
            let root_end = root_start
                + source[root_start..]
                    .find('>')
                    .context("spreadsheet style root is truncated")?
                + 1;
            source.insert_str(
                root_end,
                &format!(
                    "<numFmts count=\"{}\">{}</numFmts>",
                    format_entries.len(),
                    format_entries.join("")
                ),
            );
        }
    }
    if !font_entries.is_empty() {
        source = append_spreadsheet_style_entries(&source, "fonts", &font_entries)?;
    }
    if !xf_entries.is_empty() {
        source = append_spreadsheet_style_entries(&source, "cellXfs", &xf_entries)?;
    }
    Ok((source.into_bytes(), style_map))
}

fn append_spreadsheet_style_entries(
    source: &str,
    tag: &str,
    entries: &[String],
) -> anyhow::Result<String> {
    let closing = format!("</{tag}>");
    let position = source
        .find(&closing)
        .with_context(|| format!("spreadsheet style collection is missing: {tag}"))?;
    let mut output = source.to_string();
    output.insert_str(position, &entries.join(""));
    let opening = Regex::new(&format!(r#"<{tag}\b[^>]*\bcount=\"(\d+)\"[^>]*>"#))
        .context("compile style count regex")?;
    let capture = opening
        .captures(&output)
        .with_context(|| format!("spreadsheet style count is missing: {tag}"))?;
    let count = capture
        .get(1)
        .expect("count capture")
        .as_str()
        .parse::<usize>()?
        + entries.len();
    let range = capture.get(1).expect("count capture").range();
    output.replace_range(range, &count.to_string());
    Ok(output)
}

fn spreadsheet_font_xml(font: &SpreadsheetSourceFont) -> String {
    let mut body = String::new();
    if font.bold {
        body.push_str("<b/>");
    }
    if font.italic {
        body.push_str("<i/>");
    }
    if let Some(color) = font.color {
        body.push_str(&format!("<color rgb=\"{color:08X}\"/>"));
    }
    if let Some(size) = font.size {
        body.push_str(&format!(
            "<sz val=\"{}\"/>",
            format_spreadsheet_number(size)
        ));
    }
    if let Some(name) = &font.name {
        body.push_str(&format!("<name val=\"{}\"/>", escape_xml_attribute(name)));
    }
    format!("<font>{body}</font>")
}

fn spreadsheet_fonts_equivalent(
    left: &SpreadsheetSourceFont,
    right: &SpreadsheetSourceFont,
) -> bool {
    left.bold == right.bold
        && left.italic == right.italic
        && left.size == right.size
        && left.name == right.name
}

fn spreadsheet_xf_xml(xf: &SpreadsheetSourceXf) -> String {
    let mut attributes = format!(
        "fontId=\"{}\" fillId=\"{}\" borderId=\"{}\" numFmtId=\"{}\"",
        xf.font_id, xf.fill_id, xf.border_id, xf.num_fmt_id
    );
    if let Some(xf_id) = xf.xf_id {
        attributes.push_str(&format!(" xfId=\"{xf_id}\""));
    }
    if xf.apply_font || xf.font_id != 0 {
        attributes.push_str(" applyFont=\"1\"");
    }
    if xf.apply_fill || xf.fill_id != 0 {
        attributes.push_str(" applyFill=\"1\"");
    }
    if xf.num_fmt_id != 0 {
        attributes.push_str(" applyNumberFormat=\"1\"");
    }
    format!("<xf {attributes}/>")
}

pub(super) fn escape_xml_attribute(value: &str) -> String {
    escape_xml_text(value)
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
