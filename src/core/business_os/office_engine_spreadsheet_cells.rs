// Origin: CTOX
// License: AGPL-3.0-only

use super::*;

const SHEET_NS: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpreadsheetCellPatch {
    pub base_sha256: String,
    pub cells: Vec<SpreadsheetCellUpdate>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpreadsheetCellUpdate {
    pub sheet: String,
    pub cell: String,
    pub value: SpreadsheetCellValue,
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum SpreadsheetCellValue {
    Text(String),
    Number(f64),
    Boolean(bool),
    Formula(String),
    Clear,
}

/// Bounded, optimistic-concurrency file operation for the native harness CLI.
/// No browser, daemon store, formula evaluator or Business OS policy bypass.
pub fn patch_spreadsheet_cells(
    package: &[u8],
    patch: &SpreadsheetCellPatch,
) -> anyhow::Result<Vec<u8>> {
    ensure!(
        format!("{:x}", Sha256::digest(package)) == patch.base_sha256,
        "spreadsheet base SHA-256 mismatch; read the current file before retrying"
    );
    ensure!(
        !patch.cells.is_empty() && patch.cells.len() <= 10_000,
        "spreadsheet patch requires 1..10000 cell updates"
    );
    inspect(OfficeKind::Spreadsheet, package)?;
    let mut archive = ZipArchive::new(Cursor::new(package))?;
    let paths = spreadsheet_worksheet_paths(&mut archive)?;
    let mut replacements = BTreeMap::new();
    let mut targets = BTreeSet::new();
    for change in &patch.cells {
        cell_position(&change.cell)?;
        ensure!(
            targets.insert((&change.sheet, &change.cell)),
            "duplicate cell update"
        );
        let path = paths.get(&change.sheet).context("worksheet not found")?;
        let source = String::from_utf8(
            replacements
                .get(path)
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| read_zip_part(&mut archive, path))?,
        )?;
        let tree = roxmltree::Document::parse(&source)?;
        ensure!(
            !tree
                .descendants()
                .any(|n| n.has_tag_name((SHEET_NS, "sheetProtection"))),
            "native cell patch refuses a protected worksheet"
        );
        let existing = tree.descendants().find(|n| {
            n.has_tag_name((SHEET_NS, "c")) && n.attribute("r") == Some(change.cell.as_str())
        });
        ensure!(
            !tree.descendants().any(|n| n.has_tag_name((SHEET_NS, "f"))
                && n.attribute("t").is_some_and(|t| t != "normal")),
            "native cell patch refuses array/shared/data-table formula worksheets"
        );
        let (kind, content) = match &change.value {
            SpreadsheetCellValue::Text(value) => {
                ensure!(
                    value.encode_utf16().count() <= 32_767
                        && value
                            .chars()
                            .all(|c| matches!(c, '\t' | '\n' | '\r') || c >= ' '),
                    "invalid Excel cell text"
                );
                (
                    Some("inlineStr"),
                    format!(
                        "<is><t xml:space=\"preserve\">{}</t></is>",
                        escape_xml_attribute(value)
                    ),
                )
            }
            SpreadsheetCellValue::Number(value) => {
                ensure!(value.is_finite(), "non-finite cell number");
                (None, format!("<v>{value}</v>"))
            }
            SpreadsheetCellValue::Boolean(value) => {
                (Some("b"), format!("<v>{}</v>", u8::from(*value)))
            }
            SpreadsheetCellValue::Formula(value) => {
                let formula = value
                    .strip_prefix('=')
                    .context("formula must start with =")?;
                ensure!(
                    !formula.is_empty()
                        && formula.encode_utf16().count() <= 8192
                        && formula.chars().all(|c| c >= ' '),
                    "invalid formula text"
                );
                (None, format!("<f>{}</f>", escape_xml_attribute(formula)))
            }
            SpreadsheetCellValue::Clear => (None, String::new()),
        };
        let mut cell = existing
            .map(|n| source[n.range()].to_string())
            .unwrap_or_else(|| format!("<c r=\"{}\"/>", change.cell));
        if let Some(node) = existing {
            let start = node.range().start;
            let mut ranges: Vec<_> = node
                .children()
                .filter(|n| {
                    n.has_tag_name((SHEET_NS, "f"))
                        || n.has_tag_name((SHEET_NS, "v"))
                        || n.has_tag_name((SHEET_NS, "is"))
                })
                .map(|n| n.range())
                .collect();
            ranges.sort_by_key(|r| std::cmp::Reverse(r.start));
            for range in ranges {
                cell.replace_range(range.start - start..range.end - start, "");
            }
        }
        // Only the opening tag: text containing `t="..."` is not an attribute.
        let opening_end = cell.find('>').context("cell opening tag missing")? + 1;
        let opening = set_xml_attribute(&cell[..opening_end], "t", kind.map(str::to_string));
        cell.replace_range(..opening_end, &opening);
        // Reparse with the worksheet namespace to retain existing style metadata.
        let wrapped = format!("<root xmlns=\"{SHEET_NS}\">{cell}</root>");
        let cell_tree = roxmltree::Document::parse(&wrapped)?;
        let cell_node = cell_tree
            .root_element()
            .first_element_child()
            .context("cell missing")?;
        let filled = insert_xml_child(&wrapped, cell_node, &content)?;
        let filled_tree = roxmltree::Document::parse(&filled)?;
        let filled_node = filled_tree
            .root_element()
            .first_element_child()
            .context("cell missing")?;
        replacements.insert(
            path.clone(),
            upsert_worksheet_cell(&source, &change.cell, &filled[filled_node.range()])?
                .into_bytes(),
        );
    }
    // No cached result is invented. Consumers must recalculate this workbook.
    let workbook = String::from_utf8(read_zip_part(&mut archive, "xl/workbook.xml")?)?;
    let tree = roxmltree::Document::parse(&workbook)?;
    let calc = tree
        .root_element()
        .children()
        .find(|n| n.has_tag_name((SHEET_NS, "calcPr")));
    let updated = if let Some(calc) = calc {
        let mut updated = workbook.clone();
        let attrs = set_xml_attribute(
            &set_xml_attribute(&workbook[calc.range()], "fullCalcOnLoad", Some("1".into())),
            "forceFullCalc",
            Some("1".into()),
        );
        updated.replace_range(calc.range(), &attrs);
        updated
    } else {
        // calcPr precedes optional extension lists in the workbook schema.
        let element = "<calcPr fullCalcOnLoad=\"1\" forceFullCalc=\"1\"/>";
        if let Some(ext) = tree
            .root_element()
            .children()
            .find(|n| n.has_tag_name((SHEET_NS, "extLst")))
        {
            let mut updated = workbook.clone();
            updated.insert_str(ext.range().start, element);
            updated
        } else {
            insert_xml_child(&workbook, tree.root_element(), element)?
        }
    };
    replacements.insert("xl/workbook.xml".into(), updated.into_bytes());
    let interim = replace_package_parts(package, replacements.clone())?;
    let manifest = inspect_editor_payload(
        OfficeKind::Spreadsheet,
        &transcode_spreadsheet_to_editor_payload(&interim)?,
    )?;
    for sheet in &manifest.worksheets {
        let path = paths.get(&sheet.name).context("worksheet path missing")?;
        if let Some(bytes) = replacements.get_mut(path) {
            *bytes = update_worksheet_dimension(std::str::from_utf8(bytes)?, sheet)?.into_bytes();
        }
    }
    replace_package_parts(package, replacements)
}

pub(super) fn empty_shared_strings() -> &'static str {
    r#"<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="0" uniqueCount="0"/>"#
}

pub(super) fn update_shared_string_count(xml: &[u8], count: usize) -> anyhow::Result<Vec<u8>> {
    let source = std::str::from_utf8(xml)?;
    let tree = roxmltree::Document::parse(source)?;
    let start = tree.root_element().range().start;
    let end = start
        + source[start..]
            .find('>')
            .context("SST opening tag missing")?
        + 1;
    let opening = set_xml_attribute(&source[start..end], "count", Some(count.to_string()));
    let mut output = source.to_string();
    output.replace_range(start..end, &opening);
    Ok(output.into_bytes())
}

/// Insert inside either an ordinary or an empty XML element without rewriting
/// its attributes, namespace declarations, siblings or unrelated package data.
pub(super) fn insert_xml_child(
    source: &str,
    parent: roxmltree::Node<'_, '_>,
    child: &str,
) -> anyhow::Result<String> {
    let range = parent.range();
    let text = &source[range.clone()];
    let mut output = source.to_string();
    if text.ends_with("/>") {
        let name = text[1..]
            .split(|c: char| c.is_whitespace() || c == '/' || c == '>')
            .next()
            .context("XML element name is missing")?;
        output.replace_range(range.end - 2..range.end, &format!(">{child}</{name}>"));
    } else {
        let close = text.rfind("</").context("XML closing tag is missing")?;
        output.insert_str(range.start + close, child);
    }
    Ok(output)
}

fn cell_position(reference: &str) -> anyhow::Result<(u32, u32)> {
    let split = reference
        .find(|c: char| c.is_ascii_digit())
        .context("invalid cell reference")?;
    let (letters, digits) = reference.split_at(split);
    ensure!(
        !letters.is_empty()
            && letters.len() <= 3
            && digits.len() <= 7
            && letters.bytes().all(|c| c.is_ascii_uppercase())
            && !digits.starts_with('0')
            && digits.bytes().all(|c| c.is_ascii_digit()),
        "invalid cell reference: {reference}"
    );
    let column = letters
        .bytes()
        .fold(0u32, |n, c| n * 26 + u32::from(c - b'A' + 1));
    let row: u32 = digits.parse()?;
    ensure!(
        (1..=16_384).contains(&column) && (1..=1_048_576).contains(&row),
        "cell reference outside XLSX limits: {reference}"
    );
    Ok((row, column))
}

pub(super) fn upsert_worksheet_cell(
    source: &str,
    reference: &str,
    replacement: &str,
) -> anyhow::Result<String> {
    let (row_number, column_number) = cell_position(reference)?;
    let tree = roxmltree::Document::parse(source)?;
    let worksheet = tree.root_element();
    ensure!(
        worksheet.tag_name().name() == "worksheet"
            && worksheet.default_namespace() == Some(SHEET_NS),
        "worksheet cell writer requires the SpreadsheetML default namespace"
    );
    let sheet_data = worksheet
        .children()
        .find(|node| node.has_tag_name((SHEET_NS, "sheetData")))
        .context("worksheet sheetData is missing")?;
    let mut insert_row_at = None;
    for row in sheet_data
        .children()
        .filter(|node| node.has_tag_name((SHEET_NS, "row")))
    {
        let number: u32 = row
            .attribute("r")
            .context("worksheet row has no index")?
            .parse()?;
        if number > row_number && insert_row_at.is_none() {
            insert_row_at = Some(row.range().start);
        }
        if number != row_number {
            continue;
        }
        let mut insert_cell_at = None;
        for cell in row
            .children()
            .filter(|node| node.has_tag_name((SHEET_NS, "c")))
        {
            let existing = cell
                .attribute("r")
                .context("worksheet cell has no reference")?;
            let (existing_row, existing_column) = cell_position(existing)?;
            ensure!(existing_row == number, "worksheet cell is in the wrong row");
            if existing == reference {
                let mut output = source.to_string();
                output.replace_range(cell.range(), replacement);
                return Ok(output);
            }
            if existing_column > column_number && insert_cell_at.is_none() {
                insert_cell_at = Some(cell.range().start);
            }
        }
        if let Some(at) = insert_cell_at {
            let mut output = source.to_string();
            output.insert_str(at, replacement);
            return Ok(output);
        }
        return insert_xml_child(source, row, replacement);
    }
    let row = format!("<row r=\"{row_number}\">{replacement}</row>");
    if let Some(at) = insert_row_at {
        let mut output = source.to_string();
        output.insert_str(at, &row);
        Ok(output)
    } else {
        insert_xml_child(source, sheet_data, &row)
    }
}

/// XLSX may legally omit the SST until its first string cell is saved. Add both
/// OPC references with unique IDs; adding an unreferenced ZIP part is not enough.
pub(super) fn ensure_shared_string_relationship<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    replacements: &mut BTreeMap<String, Vec<u8>>,
) -> anyhow::Result<()> {
    let path = "xl/_rels/workbook.xml.rels";
    let bytes = replacements
        .get(path)
        .cloned()
        .map(Ok)
        .unwrap_or_else(|| read_zip_part(archive, path))?;
    let text = std::str::from_utf8(&bytes)?;
    let tree = roxmltree::Document::parse(text)?;
    let relationships: Vec<_> = tree
        .root_element()
        .children()
        .filter(|node| node.is_element())
        .collect();
    if !relationships.iter().any(|node| {
        node.attribute("Type")
            .is_some_and(|value| value.ends_with("/sharedStrings"))
    }) {
        let ids: BTreeSet<_> = relationships
            .iter()
            .filter_map(|node| node.attribute("Id"))
            .collect();
        let id = (1usize..)
            .map(|n| format!("rIdSst{n}"))
            .find(|id| !ids.contains(id.as_str()))
            .expect("unbounded fresh relationship id");
        let element = format!("<Relationship Id=\"{id}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings\" Target=\"sharedStrings.xml\"/>");
        replacements.insert(
            path.to_string(),
            insert_xml_child(text, tree.root_element(), &element)?.into_bytes(),
        );
    }
    let path = "[Content_Types].xml";
    let bytes = replacements
        .get(path)
        .cloned()
        .map(Ok)
        .unwrap_or_else(|| read_zip_part(archive, path))?;
    let text = std::str::from_utf8(&bytes)?;
    let tree = roxmltree::Document::parse(text)?;
    if !tree
        .root_element()
        .children()
        .any(|node| node.attribute("PartName") == Some("/xl/sharedStrings.xml"))
    {
        let element = r#"<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>"#;
        replacements.insert(
            path.to_string(),
            insert_xml_child(text, tree.root_element(), element)?.into_bytes(),
        );
    }
    Ok(())
}

pub(super) fn update_worksheet_dimension(
    source: &str,
    sheet: &EditorWorksheetManifest,
) -> anyhow::Result<String> {
    let positions = sheet
        .cells
        .iter()
        .map(|cell| cell_position(&cell.reference))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let first = cell_reference(
        positions.iter().map(|p| p.0).min().unwrap_or(1) - 1,
        positions.iter().map(|p| p.1).min().unwrap_or(1) - 1,
    );
    let last = cell_reference(
        positions.iter().map(|p| p.0).max().unwrap_or(1) - 1,
        positions.iter().map(|p| p.1).max().unwrap_or(1) - 1,
    );
    let reference = if first == last {
        first
    } else {
        format!("{first}:{last}")
    };
    let tree = roxmltree::Document::parse(source)?;
    let mut output = source.to_string();
    if let Some(dimension) = tree
        .root_element()
        .children()
        .find(|n| n.has_tag_name((SHEET_NS, "dimension")))
    {
        output.replace_range(
            dimension.range(),
            &set_xml_attribute(&source[dimension.range()], "ref", Some(reference)),
        );
    } else {
        let at = tree
            .root_element()
            .children()
            .filter(|n| n.is_element())
            .find(|n| !n.has_tag_name((SHEET_NS, "sheetPr")))
            .context("worksheet has no dimension insertion point")?
            .range()
            .start;
        output.insert_str(at, &format!("<dimension ref=\"{reference}\"/>"));
    }
    Ok(output)
}

/// Reconcile the workbook's sheet directory before cell/layout export. Existing
/// worksheet parts keep their stable relationship and all escrow content;
/// newly created sheets start empty. Removed sheets lose their directory entry,
/// worksheet and worksheet-rel part, never another sheet's shared dependencies.
pub(super) fn reconcile_worksheet_structure(
    package: &[u8],
    before: &EditorPayloadManifest,
    after: &EditorPayloadManifest,
) -> anyhow::Result<Option<Vec<u8>>> {
    ensure!(
        !after.worksheets.is_empty(),
        "a workbook must contain a worksheet"
    );
    let mut names = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for sheet in &after.worksheets {
        ensure!(
            !sheet.name.is_empty()
                && sheet.name.chars().count() <= 31
                && !sheet.name.contains(['[', ']', ':', '*', '?', '/', '\\'])
                && !sheet.name.starts_with('\'')
                && !sheet.name.ends_with('\''),
            "invalid worksheet name: {}",
            sheet.name
        );
        ensure!(
            names.insert(sheet.name.to_lowercase()) && ids.insert(sheet.sheet_id),
            "duplicate worksheet name or identity"
        );
        ensure!(
            matches!(
                sheet.visibility.as_str(),
                "visible" | "hidden" | "very_hidden"
            ),
            "invalid worksheet visibility"
        );
    }
    ensure!(
        after.worksheets.iter().any(|s| s.visibility == "visible"),
        "a workbook must contain a visible worksheet"
    );
    if before.worksheets.len() == after.worksheets.len()
        && before
            .worksheets
            .iter()
            .zip(&after.worksheets)
            .all(|(a, b)| {
                a.sheet_id == b.sheet_id && a.name == b.name && a.visibility == b.visibility
            })
    {
        return Ok(None);
    }
    let mut archive = ZipArchive::new(Cursor::new(package))?;
    let paths = spreadsheet_worksheet_paths(&mut archive)?;
    let workbook = String::from_utf8(read_zip_part(&mut archive, "xl/workbook.xml")?)?;
    let tree = roxmltree::Document::parse(&workbook)?;
    let sheets = tree
        .root_element()
        .children()
        .find(|n| n.has_tag_name((SHEET_NS, "sheets")))
        .context("workbook sheets are missing")?;
    let old_entries = sheets
        .children()
        .filter(|n| n.has_tag_name((SHEET_NS, "sheet")))
        .map(|n| {
            Ok((
                n.attribute("sheetId")
                    .context("sheetId missing")?
                    .parse::<u32>()?,
                n,
            ))
        })
        .collect::<anyhow::Result<BTreeMap<_, _>>>()?;
    let relationships =
        String::from_utf8(read_zip_part(&mut archive, "xl/_rels/workbook.xml.rels")?)?;
    let rel_tree = roxmltree::Document::parse(&relationships)?;
    let mut relationship_ids: BTreeSet<String> = rel_tree
        .root_element()
        .children()
        .filter_map(|n| n.attribute("Id").map(str::to_string))
        .collect();
    let mut package_paths = BTreeSet::new();
    for i in 0..archive.len() {
        package_paths.insert(archive.by_index(i)?.name().to_string());
    }
    let mut replacements = BTreeMap::new();
    let mut removals = BTreeSet::new();
    let mut removed_relationships = BTreeSet::new();
    let mut added_relationships = String::new();
    let mut added_types = String::new();
    let mut new_sheets = String::from("<sheets>");
    for sheet in &after.worksheets {
        let relation = if let Some(old) = old_entries.get(&sheet.sheet_id) {
            old.attributes()
                .find(|a| a.name() == "id")
                .context("worksheet relationship missing")?
                .value()
                .to_string()
        } else {
            let relation = (1usize..)
                .map(|n| format!("rIdCtoxSheet{n}"))
                .find(|id| !relationship_ids.contains(id))
                .expect("fresh relationship ID");
            relationship_ids.insert(relation.clone());
            let path = (1usize..)
                .map(|n| format!("xl/worksheets/ctox-sheet-{n}.xml"))
                .find(|path| !package_paths.contains(path))
                .expect("fresh worksheet path");
            package_paths.insert(path.clone());
            let target = path.strip_prefix("xl/").expect("worksheet path prefix");
            added_relationships.push_str(&format!(r#"<Relationship Id="{relation}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="{target}"/>"#));
            added_types.push_str(&format!(r#"<Override PartName="/{path}" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>"#));
            replacements.insert(path, format!(r#"<worksheet xmlns="{SHEET_NS}"><dimension ref="A1"/><sheetViews><sheetView workbookViewId="0"/></sheetViews><sheetFormatPr defaultRowHeight="15"/><sheetData/></worksheet>"#).into_bytes());
            relation
        };
        let visibility = match sheet.visibility.as_str() {
            "hidden" => " state=\"hidden\"",
            "very_hidden" => " state=\"veryHidden\"",
            _ => "",
        };
        new_sheets.push_str(&format!(r#"<sheet xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" name="{}" sheetId="{}" r:id="{}"{visibility}/>"#,
            escape_xml_attribute(&sheet.name),sheet.sheet_id,escape_xml_attribute(&relation)));
    }
    new_sheets.push_str("</sheets>");
    for old in &before.worksheets {
        if ids.contains(&old.sheet_id) {
            continue;
        }
        let path = paths
            .get(&old.name)
            .context("removed worksheet path missing")?;
        removals.insert(path.clone());
        let (dir, file) = path
            .rsplit_once('/')
            .context("worksheet directory missing")?;
        removals.insert(format!("{dir}/_rels/{file}.rels"));
        let entry = old_entries
            .get(&old.sheet_id)
            .context("removed sheet entry missing")?;
        removed_relationships.insert(
            entry
                .attributes()
                .find(|a| a.name() == "id")
                .context("removed worksheet relationship missing")?
                .value()
                .to_string(),
        );
    }
    let mut updated_workbook = workbook.clone();
    updated_workbook.replace_range(sheets.range(), &new_sheets);
    // Removed/reordered worksheets invalidate positional calculation chains.
    for rel in rel_tree
        .root_element()
        .children()
        .filter(|n| n.is_element())
    {
        if rel
            .attribute("Type")
            .is_some_and(|v| v.ends_with("/calcChain"))
        {
            removed_relationships.insert(
                rel.attribute("Id")
                    .context("calculation chain ID missing")?
                    .to_string(),
            );
            let target = rel
                .attribute("Target")
                .context("calculation chain target missing")?;
            ensure!(!target.contains(".."), "unsafe calculation chain target");
            removals.insert(if target.starts_with("/xl/") {
                target.trim_start_matches('/').to_string()
            } else if target.starts_with("xl/") {
                target.to_string()
            } else {
                format!("xl/{target}")
            });
        }
    }
    let updated_tree = roxmltree::Document::parse(&updated_workbook)?;
    let calc = updated_tree
        .root_element()
        .children()
        .find(|n| n.has_tag_name((SHEET_NS, "calcPr")));
    if let Some(calc) = calc {
        let range = calc.range();
        let updated = set_xml_attribute(
            &set_xml_attribute(
                &updated_workbook[range.clone()],
                "fullCalcOnLoad",
                Some("1".into()),
            ),
            "forceFullCalc",
            Some("1".into()),
        );
        updated_workbook.replace_range(range, &updated);
    }
    replacements.insert("xl/workbook.xml".into(), updated_workbook.into_bytes());
    let mut updated_rels = relationships.clone();
    let mut ranges: Vec<_> = rel_tree
        .root_element()
        .children()
        .filter(|n| {
            n.attribute("Id")
                .is_some_and(|id| removed_relationships.contains(id))
        })
        .map(|n| n.range())
        .collect();
    ranges.sort_by_key(|r| std::cmp::Reverse(r.start));
    for range in ranges {
        updated_rels.replace_range(range, "");
    }
    let updated_tree = roxmltree::Document::parse(&updated_rels)?;
    let updated_rels = insert_xml_child(
        &updated_rels,
        updated_tree.root_element(),
        &added_relationships,
    )?;
    replacements.insert(
        "xl/_rels/workbook.xml.rels".into(),
        updated_rels.into_bytes(),
    );
    let types = String::from_utf8(read_zip_part(&mut archive, "[Content_Types].xml")?)?;
    let tree = roxmltree::Document::parse(&types)?;
    let mut ranges: Vec<_> = tree
        .root_element()
        .children()
        .filter(|n| {
            n.attribute("PartName")
                .is_some_and(|p| removals.contains(p.trim_start_matches('/')))
        })
        .map(|n| n.range())
        .collect();
    let mut updated_types = types.clone();
    ranges.sort_by_key(|r| std::cmp::Reverse(r.start));
    for range in ranges {
        updated_types.replace_range(range, "");
    }
    let tree = roxmltree::Document::parse(&updated_types)?;
    let updated_types = insert_xml_child(&updated_types, tree.root_element(), &added_types)?;
    replacements.insert("[Content_Types].xml".into(), updated_types.into_bytes());
    drop(archive);
    Ok(Some(replace_package_parts_removing(
        package,
        replacements,
        &removals,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_cell_batch_is_typed_guarded_and_preserves_escrow() {
        let source = delimited_text_to_xlsx(b"Keep,10\n").unwrap();
        let name = editor_manifest(&source).worksheets[0].name.clone();
        let patch: SpreadsheetCellPatch = serde_json::from_value(serde_json::json!({
            "base_sha256":format!("{:x}",Sha256::digest(&source)),
            "cells":[
                {"sheet":name,"cell":"B1","value":{"type":"number","value":20}},
                {"sheet":name,"cell":"C12","value":{"type":"text","value":"  & =literal  "}},
                {"sheet":name,"cell":"D12","value":{"type":"formula","value":"=SUM(B1,22)"}},
                {"sheet":name,"cell":"E12","value":{"type":"boolean","value":true}}
            ]
        }))
        .unwrap();
        let result = patch_spreadsheet_cells(&source, &patch).unwrap();
        let cells = &editor_manifest(&result).worksheets[0].cells;
        assert!(cells
            .iter()
            .any(|c| c.reference == "A1" && c.display == "Keep"));
        assert!(cells
            .iter()
            .any(|c| c.reference == "B1" && c.display == "20"));
        assert!(cells
            .iter()
            .any(|c| c.reference == "C12" && c.display == "  & =literal  "));
        assert!(cells
            .iter()
            .any(|c| c.reference == "D12" && c.formula.as_deref() == Some("=SUM(B1,22)")));
        let mut original_zip = ZipArchive::new(Cursor::new(&source)).unwrap();
        let mut updated_zip = ZipArchive::new(Cursor::new(&result)).unwrap();
        assert_eq!(
            read_zip_part(&mut original_zip, "xl/styles.xml").unwrap(),
            read_zip_part(&mut updated_zip, "xl/styles.xml").unwrap()
        );
        let mut stale = patch;
        stale.base_sha256 = "0".repeat(64);
        assert!(patch_spreadsheet_cells(&source, &stale)
            .unwrap_err()
            .to_string()
            .contains("SHA-256 mismatch"));
    }

    #[test]
    fn native_cell_batch_rejects_duplicate_and_unsafe_values() {
        let source = delimited_text_to_xlsx(b"Keep\n").unwrap();
        let name = editor_manifest(&source).worksheets[0].name.clone();
        for updates in [
            serde_json::json!([
                {"sheet":name,"cell":"A1","value":{"type":"clear"}},
                {"sheet":name,"cell":"A1","value":{"type":"clear"}}
            ]),
            serde_json::json!([{"sheet":name,"cell":"A1","value":{"type":"text","value":"\u{0001}"}}]),
            serde_json::json!([{"sheet":name,"cell":"XFE1","value":{"type":"clear"}}]),
            serde_json::json!([{"sheet":"missing","cell":"A1","value":{"type":"clear"}}]),
            serde_json::json!([{"sheet":name,"cell":"A1","value":{"type":"formula","value":"SUM(1,2)"}}]),
        ] {
            let patch = serde_json::from_value(serde_json::json!({
                "base_sha256":format!("{:x}",Sha256::digest(&source)),"cells":updates
            }))
            .unwrap();
            assert!(patch_spreadsheet_cells(&source, &patch).is_err());
        }
    }

    #[test]
    fn empty_sheet_data_and_self_closing_cells_accept_first_content() {
        let source = format!(r#"<worksheet xmlns="{SHEET_NS}"><sheetData/></worksheet>"#);
        let first = upsert_worksheet_cell(&source, "C12", r#"<c r="C12"><v>12</v></c>"#).unwrap();
        let next = upsert_worksheet_cell(&first, "A2", r#"<c r="A2"/>"#).unwrap();
        let next = upsert_worksheet_cell(&next, "A2", r#"<c r="A2"><v>7</v></c>"#).unwrap();
        let tree = roxmltree::Document::parse(&next).unwrap();
        let cells: Vec<_> = tree
            .descendants()
            .filter(|n| n.has_tag_name((SHEET_NS, "c")))
            .map(|n| n.attribute("r").unwrap())
            .collect();
        assert_eq!(cells, ["A2", "C12"]);
        assert!(next.contains("<v>7</v>"));
    }

    #[test]
    fn insertion_preserves_row_metadata_and_numeric_cell_order() {
        let source = format!(
            r#"<worksheet xmlns="{SHEET_NS}"><sheetData><row r="3" ht="24" customHeight="1"><c r="A3"/><c r="AA3"/></row><row r="7"/></sheetData><extLst/></worksheet>"#
        );
        let next = upsert_worksheet_cell(&source, "Z3", r#"<c r="Z3"><v>4</v></c>"#).unwrap();
        let next = upsert_worksheet_cell(&next, "B7", r#"<c r="B7"/>"#).unwrap();
        assert!(next.contains(r#"<row r="3" ht="24" customHeight="1">"#));
        assert!(next.find("Z3").unwrap() < next.find("AA3").unwrap());
        assert!(next.contains(r#"<row r="7"><c r="B7"/></row>"#));
        assert!(next.contains("<extLst/>"));
    }

    #[test]
    fn cell_positions_are_bounded_and_unambiguous() {
        assert_eq!(cell_position("XFD1048576").unwrap(), (1_048_576, 16_384));
        for bad in [
            "A0",
            "A01",
            "A1048577",
            "XFE1",
            "AAAA1",
            "$A$1",
            "a1",
            "1",
            "A1x",
            "A99999999999999999999999",
        ] {
            assert!(cell_position(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn empty_shared_string_table_can_grow_without_losing_whitespace() {
        let output = replace_changed_shared_strings(
            empty_shared_strings().as_bytes(),
            &[],
            &["  first & last  ".to_string()],
        )
        .unwrap();
        assert_eq!(
            parse_ooxml_shared_strings(&output).unwrap(),
            ["  first & last  "]
        );
        assert!(String::from_utf8(output)
            .unwrap()
            .contains("uniqueCount=\"1\""));
    }

    fn editor_manifest(package: &[u8]) -> EditorPayloadManifest {
        inspect_editor_payload(
            OfficeKind::Spreadsheet,
            &transcode_spreadsheet_to_editor_payload(package).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn workbook_add_rename_reorder_edit_and_remove_roundtrip() {
        let source = delimited_text_to_xlsx(b"Keep,10\n").unwrap();
        let original = editor_manifest(&source);
        let mut desired = original.clone();
        desired.worksheets[0].name = "Renamed".into();
        let mut added = desired.worksheets[0].clone();
        added.name = "Second & Team".into();
        added.sheet_id = 2;
        desired.worksheets.insert(0, added);
        let target = reconcile_worksheet_structure(&source, &original, &desired)
            .unwrap()
            .unwrap();
        let mut zip = ZipArchive::new(Cursor::new(&target)).unwrap();
        let paths = spreadsheet_worksheet_paths(&mut zip).unwrap();
        let new_path = paths["Second & Team"].clone();
        let mut changes = BTreeMap::new();
        changes.insert(new_path.clone(), format!(r#"<worksheet xmlns="{SHEET_NS}"><dimension ref="A10:B10"/><sheetData><row r="10"><c r="A10" t="inlineStr"><is><t>New cell</t></is></c><c r="B10"><f>SUM(20,22)</f><v>42</v></c></row></sheetData></worksheet>"#).into_bytes());
        let target = replace_package_parts(&target, changes).unwrap();
        let payload = transcode_spreadsheet_to_editor_payload(&target).unwrap();
        let exported = export(OfficeKind::Spreadsheet, &payload, Some(&source)).unwrap();
        let actual = editor_manifest(&exported.bytes);
        assert_eq!(
            actual
                .worksheets
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["Second & Team", "Renamed"]
        );
        assert!(actual.worksheets[0]
            .cells
            .iter()
            .any(|c| c.reference == "A10" && c.display == "New cell"));
        assert!(actual.worksheets[0]
            .cells
            .iter()
            .any(|c| c.reference == "B10"
                && c.formula.as_deref() == Some("=SUM(20,22)")
                && c.display == "42"));
        assert_eq!(actual.worksheets[1].cells, original.worksheets[0].cells);
        let mut one_sheet = actual.clone();
        one_sheet.worksheets.remove(0);
        let target = reconcile_worksheet_structure(&exported.bytes, &actual, &one_sheet)
            .unwrap()
            .unwrap();
        let payload = transcode_spreadsheet_to_editor_payload(&target).unwrap();
        let final_package =
            export(OfficeKind::Spreadsheet, &payload, Some(&exported.bytes)).unwrap();
        assert_eq!(editor_manifest(&final_package.bytes).worksheets.len(), 1);
        let mut zip = ZipArchive::new(Cursor::new(&final_package.bytes)).unwrap();
        assert!(
            zip.by_name(&new_path).is_err(),
            "deleted worksheet bytes retained"
        );
    }

    #[test]
    fn worksheet_structure_rejects_invalid_and_all_hidden_workbooks() {
        let source = delimited_text_to_xlsx(b"Keep\n").unwrap();
        let before = editor_manifest(&source);
        let mut changed = before.clone();
        changed.worksheets[0].visibility = "hidden".into();
        assert!(reconcile_worksheet_structure(&source, &before, &changed).is_err());
        changed.worksheets[0].visibility = "visible".into();
        changed.worksheets[0].name = "Not/Valid".into();
        assert!(reconcile_worksheet_structure(&source, &before, &changed).is_err());
        changed = before.clone();
        changed.worksheets.push(changed.worksheets[0].clone());
        assert!(reconcile_worksheet_structure(&source, &before, &changed).is_err());
    }

    #[test]
    fn first_string_into_existing_numeric_row_and_new_row_exports() {
        // Numeric-only CSV produces a valid, empty SST. A previously unseen
        // cell must append its first string rather than require an old cell.
        let source = delimited_text_to_xlsx(b"1,2\n").unwrap();
        let target = delimited_text_to_xlsx(b"1,first\n3,second\n").unwrap();
        let payload = transcode_spreadsheet_to_editor_payload(&target).unwrap();
        let result = export(OfficeKind::Spreadsheet, &payload, Some(&source)).unwrap();
        let actual = editor_manifest(&result.bytes);
        assert!(actual.worksheets[0]
            .cells
            .iter()
            .any(|c| c.reference == "B1" && c.display == "first"));
        assert!(actual.worksheets[0]
            .cells
            .iter()
            .any(|c| c.reference == "B2" && c.display == "second"));
    }
}
