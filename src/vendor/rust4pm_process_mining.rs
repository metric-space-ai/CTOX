// Origin: adapted from rust4pm process_mining concepts
// Source: https://github.com/aarkue/rust4pm/tree/main/process_mining
// License: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Event {
    pub activity: String,
    pub timestamp: Option<String>,
    pub attributes_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Trace {
    pub case_id: String,
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct EventLog {
    pub traces: Vec<Trace>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct EventLogActivityProjection {
    pub activities: Vec<String>,
    pub act_to_index: BTreeMap<String, usize>,
    pub traces: Vec<(Vec<usize>, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DirectlyFollowsGraph {
    pub activities: BTreeMap<String, u64>,
    pub edges: BTreeMap<(String, String), u64>,
    pub start_activities: BTreeSet<String>,
    pub end_activities: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PetriTransition {
    pub transition_id: String,
    pub label: Option<String>,
    pub is_silent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PetriArc {
    pub arc_id: String,
    pub from_node_id: String,
    pub from_node_kind: PetriNodeKind,
    pub to_node_id: String,
    pub to_node_kind: PetriNodeKind,
    pub weight: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PetriNodeKind {
    Place,
    Transition,
}

impl PetriNodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PetriNodeKind::Place => "place",
            PetriNodeKind::Transition => "transition",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PetriNet {
    pub places: BTreeSet<String>,
    pub transitions: BTreeMap<String, PetriTransition>,
    pub arcs: Vec<PetriArc>,
    pub initial_marking: BTreeMap<String, u64>,
    pub final_markings: Vec<BTreeMap<String, u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TokenReplayResult {
    pub produced: u64,
    pub consumed: u64,
    pub missing: u64,
    pub remaining: u64,
    pub trace_count: u64,
}

impl TokenReplayResult {
    pub fn fitness(&self) -> Option<f64> {
        if self.consumed == 0 || self.produced == 0 {
            return None;
        }
        Some(
            0.5 * (1.0 - (self.missing as f64 / self.consumed as f64))
                + 0.5 * (1.0 - (self.remaining as f64 / self.produced as f64)),
        )
    }
}

pub fn activity_projection(log: &EventLog) -> EventLogActivityProjection {
    let mut act_to_index = BTreeMap::new();
    let mut activities = Vec::new();
    let mut variants: BTreeMap<Vec<usize>, u64> = BTreeMap::new();

    for trace in &log.traces {
        let mut projected = Vec::with_capacity(trace.events.len());
        for event in &trace.events {
            let next_index = activities.len();
            let index = *act_to_index
                .entry(event.activity.clone())
                .or_insert_with(|| {
                    activities.push(event.activity.clone());
                    next_index
                });
            projected.push(index);
        }
        *variants.entry(projected).or_default() += 1;
    }

    EventLogActivityProjection {
        activities,
        act_to_index,
        traces: variants.into_iter().collect(),
    }
}

pub fn discover_dfg(log: &EventLog) -> DirectlyFollowsGraph {
    let mut dfg = DirectlyFollowsGraph::default();
    for trace in &log.traces {
        let mut last: Option<&str> = None;
        for event in &trace.events {
            *dfg.activities.entry(event.activity.clone()).or_default() += 1;
            if let Some(previous) = last {
                *dfg.edges
                    .entry((previous.to_string(), event.activity.clone()))
                    .or_default() += 1;
            } else {
                dfg.start_activities.insert(event.activity.clone());
            }
            last = Some(event.activity.as_str());
        }
        if let Some(last) = last {
            dfg.end_activities.insert(last.to_string());
        }
    }
    dfg
}

pub fn discover_petri_from_dfg(dfg: &DirectlyFollowsGraph) -> PetriNet {
    let mut net = PetriNet::default();
    net.places.insert("p_start".to_string());
    net.places.insert("p_end".to_string());
    net.initial_marking.insert("p_start".to_string(), 1);
    net.final_markings
        .push(BTreeMap::from([("p_end".to_string(), 1)]));

    for activity in dfg.activities.keys() {
        let transition_id = transition_id(activity);
        net.transitions.insert(
            transition_id.clone(),
            PetriTransition {
                transition_id,
                label: Some(activity.clone()),
                is_silent: false,
            },
        );
    }

    for activity in &dfg.start_activities {
        net.arcs.push(PetriArc {
            arc_id: arc_id("p_start", &transition_id(activity)),
            from_node_id: "p_start".to_string(),
            from_node_kind: PetriNodeKind::Place,
            to_node_id: transition_id(activity),
            to_node_kind: PetriNodeKind::Transition,
            weight: 1,
        });
    }

    for activity in &dfg.end_activities {
        net.arcs.push(PetriArc {
            arc_id: arc_id(&transition_id(activity), "p_end"),
            from_node_id: transition_id(activity),
            from_node_kind: PetriNodeKind::Transition,
            to_node_id: "p_end".to_string(),
            to_node_kind: PetriNodeKind::Place,
            weight: 1,
        });
    }

    for (from, to) in dfg.edges.keys() {
        let place_id = format!("p_df_{}_{}", stable_id(from), stable_id(to));
        net.places.insert(place_id.clone());
        net.arcs.push(PetriArc {
            arc_id: arc_id(&transition_id(from), &place_id),
            from_node_id: transition_id(from),
            from_node_kind: PetriNodeKind::Transition,
            to_node_id: place_id.clone(),
            to_node_kind: PetriNodeKind::Place,
            weight: 1,
        });
        net.arcs.push(PetriArc {
            arc_id: arc_id(&place_id, &transition_id(to)),
            from_node_id: place_id,
            from_node_kind: PetriNodeKind::Place,
            to_node_id: transition_id(to),
            to_node_kind: PetriNodeKind::Transition,
            weight: 1,
        });
    }

    net
}

pub fn token_replay(net: &PetriNet, projection: &EventLogActivityProjection) -> TokenReplayResult {
    let mut result = TokenReplayResult::default();
    let transition_by_activity = net
        .transitions
        .values()
        .filter_map(|transition| {
            transition
                .label
                .as_ref()
                .map(|label| (label.clone(), transition.transition_id.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let input_arcs = arcs_by_transition(net, PetriNodeKind::Place, PetriNodeKind::Transition);
    let output_arcs = arcs_by_transition(net, PetriNodeKind::Transition, PetriNodeKind::Place);
    let final_marking = net.final_markings.first().cloned().unwrap_or_default();

    for (trace, frequency) in &projection.traces {
        result.trace_count += frequency;
        let mut marking = net.initial_marking.clone();
        result.produced += marking.values().sum::<u64>() * frequency;

        for activity_index in trace {
            let Some(activity) = projection.activities.get(*activity_index) else {
                continue;
            };
            let Some(transition_id) = transition_by_activity.get(activity) else {
                result.missing += frequency;
                continue;
            };

            for (place_id, weight) in input_arcs.get(transition_id).into_iter().flatten() {
                result.consumed += weight * frequency;
                let tokens = marking.entry(place_id.clone()).or_default();
                if *tokens < *weight {
                    result.missing += (*weight - *tokens) * frequency;
                    *tokens = 0;
                } else {
                    *tokens -= *weight;
                }
            }
            for (place_id, weight) in output_arcs.get(transition_id).into_iter().flatten() {
                result.produced += weight * frequency;
                *marking.entry(place_id.clone()).or_default() += *weight;
            }
        }

        for (place_id, weight) in &final_marking {
            result.consumed += weight * frequency;
            let tokens = marking.entry(place_id.clone()).or_default();
            if *tokens < *weight {
                result.missing += (*weight - *tokens) * frequency;
                *tokens = 0;
            } else {
                *tokens -= *weight;
            }
        }
        result.remaining += marking.values().sum::<u64>() * frequency;
    }

    result
}

pub fn petri_deadlock_suspects(net: &PetriNet) -> Vec<String> {
    let outgoing_places = net
        .arcs
        .iter()
        .filter(|arc| arc.from_node_kind == PetriNodeKind::Place)
        .map(|arc| arc.from_node_id.as_str())
        .collect::<BTreeSet<_>>();
    let final_places = net
        .final_markings
        .iter()
        .flat_map(|marking| marking.keys().map(String::as_str))
        .collect::<BTreeSet<_>>();
    net.places
        .iter()
        .filter(|place| {
            !outgoing_places.contains(place.as_str()) && !final_places.contains(place.as_str())
        })
        .cloned()
        .collect()
}

pub fn petri_to_dot(net: &PetriNet) -> String {
    let mut out = String::from("digraph ctox_petri_net {\n  rankdir=LR;\n");
    for place in &net.places {
        out.push_str(&format!("  \"{place}\" [shape=circle];\n"));
    }
    for transition in net.transitions.values() {
        let label = transition.label.as_deref().unwrap_or("tau");
        out.push_str(&format!(
            "  \"{}\" [shape=box,label=\"{}\"];\n",
            transition.transition_id,
            label.replace('"', "'")
        ));
    }
    for arc in &net.arcs {
        out.push_str(&format!(
            "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
            arc.from_node_id, arc.to_node_id, arc.weight
        ));
    }
    out.push_str("}\n");
    out
}

fn arcs_by_transition(
    net: &PetriNet,
    from_kind: PetriNodeKind,
    to_kind: PetriNodeKind,
) -> BTreeMap<String, Vec<(String, u64)>> {
    let mut arcs = BTreeMap::<String, Vec<(String, u64)>>::new();
    for arc in &net.arcs {
        if arc.from_node_kind == from_kind && arc.to_node_kind == to_kind {
            let (transition_id, place_id) = match from_kind {
                PetriNodeKind::Place => (arc.to_node_id.clone(), arc.from_node_id.clone()),
                PetriNodeKind::Transition => (arc.from_node_id.clone(), arc.to_node_id.clone()),
            };
            arcs.entry(transition_id)
                .or_default()
                .push((place_id, arc.weight));
        }
    }
    arcs
}

fn transition_id(activity: &str) -> String {
    format!("t_{}", stable_id(activity))
}

fn arc_id(from: &str, to: &str) -> String {
    format!("a_{}_{}", stable_id(from), stable_id(to))
}

fn stable_id(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "unnamed".to_string()
    } else {
        trimmed
    }
}

pub fn reachable_places(net: &PetriNet) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::new();
    for place in net.initial_marking.keys() {
        if seen.insert(place.clone()) {
            queue.push_back(place.clone());
        }
    }

    while let Some(node) = queue.pop_front() {
        for arc in net.arcs.iter().filter(|arc| arc.from_node_id == node) {
            if seen.insert(arc.to_node_id.clone()) {
                queue.push_back(arc.to_node_id.clone());
            }
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_and_replays_simple_sequence() {
        let log = EventLog {
            traces: vec![Trace {
                case_id: "case-1".to_string(),
                events: vec![
                    Event {
                        activity: "A".to_string(),
                        timestamp: None,
                        attributes_json: "{}".to_string(),
                    },
                    Event {
                        activity: "B".to_string(),
                        timestamp: None,
                        attributes_json: "{}".to_string(),
                    },
                ],
            }],
        };
        let projection = activity_projection(&log);
        let dfg = discover_dfg(&log);
        let net = discover_petri_from_dfg(&dfg);
        let replay = token_replay(&net, &projection);

        assert_eq!(projection.traces.len(), 1);
        assert!(dfg.edges.contains_key(&("A".to_string(), "B".to_string())));
        assert_eq!(replay.missing, 0);
    }
}
