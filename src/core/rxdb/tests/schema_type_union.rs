use rxdb::custom_index::get_index_meta;
use rxdb::rx_schema::validate_write_document;
use rxdb::types::schema::RxJsonSchema;
use serde_json::json;

fn schema() -> RxJsonSchema {
    serde_json::from_value(json!({
        "version": 0, "primaryKey": "id", "type": "object",
        "properties": {
            "id": {"type":"string", "maxLength":64},
            "active_task_id": {"type":["string","null"], "maxLength":8},
            "mixed": {"type":["integer","boolean"]}
        },
        "required": ["id", "active_task_id"]
    }))
    .unwrap()
}

#[test]
fn nullable_required_fields_are_present_but_not_untyped() {
    let schema = schema();
    for value in [json!(null), json!("task-a")] {
        assert!(
            validate_write_document(&schema, "id", &json!({"id":"a", "active_task_id":value}))
                .is_ok()
        );
    }
    for value in [
        json!(42),
        json!(false),
        json!({}),
        json!([]),
        json!("too-long-task-id"),
    ] {
        assert!(
            validate_write_document(&schema, "id", &json!({"id":"a", "active_task_id":value}))
                .is_err()
        );
    }
    assert!(validate_write_document(&schema, "id", &json!({"id":"a"})).is_err());
}

#[test]
fn multiple_types_validate_each_member_without_allowing_arbitrary_values() {
    let schema = schema();
    for value in [json!(7), json!(false)] {
        assert!(validate_write_document(
            &schema,
            "id",
            &json!({"id":"a", "active_task_id":null, "mixed":value})
        )
        .is_ok());
    }
    for value in [json!(1.5), json!("7"), json!([])] {
        assert!(validate_write_document(
            &schema,
            "id",
            &json!({"id":"a", "active_task_id":null, "mixed":value})
        )
        .is_err());
    }
}

#[test]
fn mixed_type_indexes_return_an_error_instead_of_panicking() {
    assert!(get_index_meta(&schema(), &["active_task_id".to_owned()]).is_err());
    assert!(get_index_meta(&schema(), &["id".to_owned()]).is_ok());
}
