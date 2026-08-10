use std::cmp::Ordering;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VerticalSlashSelection {
    pub vertical_positions: Vec<usize>,
    pub slash_distances: Vec<usize>,
}

fn descending_indices(values: &[f32], count: usize) -> Result<Vec<usize>, String> {
    if count == 0 || count > values.len() || values.iter().any(|value| value.is_nan()) {
        return Err("structured selector top-k shape or value mismatch".to_owned());
    }
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        values[right]
            .partial_cmp(&values[left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(count);
    Ok(indices)
}

pub(crate) fn causal_f32_softmax_rows(
    scores: &[f32],
    last_queries: usize,
    context: usize,
) -> Result<Vec<f32>, String> {
    if last_queries == 0
        || last_queries > context
        || scores.len() != last_queries * context
        || scores.iter().any(|value| !value.is_finite())
    {
        return Err("structured selector score shape or value mismatch".to_owned());
    }
    let query_start = context - last_queries;
    let mut probabilities = vec![0.0_f32; scores.len()];
    for row in 0..last_queries {
        let visible = query_start + row + 1;
        let source = &scores[row * context..row * context + visible];
        let maximum = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let destination = &mut probabilities[row * context..row * context + visible];
        let mut denominator = 0.0_f32;
        for (output, score) in destination.iter_mut().zip(source) {
            *output = (*score - maximum).exp();
            denominator += *output;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("structured selector softmax denominator is invalid".to_owned());
        }
        for output in destination {
            *output /= denominator;
        }
    }
    Ok(probabilities)
}

pub(crate) fn vertical_slash_selection(
    probabilities: &[f32],
    last_queries: usize,
    context: usize,
    vertical_size: usize,
    slash_size: usize,
) -> Result<VerticalSlashSelection, String> {
    if last_queries == 0
        || last_queries > context
        || probabilities.len() != last_queries * context
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("structured selector probability shape or value mismatch".to_owned());
    }
    let vertical_size = vertical_size.clamp(30.min(context), context);
    let slash_size = slash_size.clamp(50.min(context), context);
    let query_start = context - last_queries;
    let mut vertical_scores = vec![0.0_f32; context];
    let mut slash_scores = vec![0.0_f32; context];
    for row in 0..last_queries {
        let query_position = query_start + row;
        for key_position in 0..=query_position {
            let probability = probabilities[row * context + key_position];
            vertical_scores[key_position] += probability;
            slash_scores[query_position - key_position] += probability;
        }
        if probabilities[row * context + query_position + 1..(row + 1) * context]
            .iter()
            .any(|value| *value != 0.0)
        {
            return Err("structured selector probability row is not causal".to_owned());
        }
    }
    for score in vertical_scores.iter_mut().take(30.min(context)) {
        *score = f32::INFINITY;
    }
    for score in slash_scores.iter_mut().take(100.min(context)) {
        *score = f32::INFINITY;
    }
    let mut vertical_positions = descending_indices(&vertical_scores, vertical_size)?;
    // MInference ranks the output of `sum_all_diagonal_matrix` before converting
    // source diagonal index `i` to distance `(context - 1) - i`. Preserve the
    // frozen lower-original-index tie rule in that source index space.
    let slash_source_scores = slash_scores.iter().rev().copied().collect::<Vec<_>>();
    let mut slash_distances = descending_indices(&slash_source_scores, slash_size)?
        .into_iter()
        .map(|source_index| context - 1 - source_index)
        .collect::<Vec<_>>();
    vertical_positions.sort_unstable();
    slash_distances.sort_unstable();
    Ok(VerticalSlashSelection {
        vertical_positions,
        slash_distances,
    })
}

pub(crate) fn selected_positions_for_query(
    query_position: usize,
    selection: &VerticalSlashSelection,
) -> Result<Vec<usize>, String> {
    if selection
        .vertical_positions
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
        || selection
            .slash_distances
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err("structured selector indices are not strictly ordered".to_owned());
    }
    let mut positions = selection
        .vertical_positions
        .iter()
        .copied()
        .filter(|position| *position <= query_position)
        .collect::<Vec<_>>();
    positions.extend(
        selection
            .slash_distances
            .iter()
            .copied()
            .filter(|distance| *distance <= query_position)
            .map(|distance| query_position - distance),
    );
    positions.sort_unstable();
    positions.dedup();
    if positions.is_empty()
        || positions
            .last()
            .is_some_and(|value| *value > query_position)
    {
        return Err("structured selector produced an empty or noncausal union".to_owned());
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_softmax_masks_future_and_preserves_f32_rows() {
        let scores = vec![
            0.0, 1.0, 2.0, 3.0, 900.0, // absolute query 3
            1.0, 1.0, 1.0, 1.0, 1.0, // absolute query 4
        ];
        let probabilities = causal_f32_softmax_rows(&scores, 2, 5).unwrap();
        for row in 0..2 {
            let visible = 4 + row;
            let sum = probabilities[row * 5..row * 5 + visible]
                .iter()
                .sum::<f32>();
            assert!((sum - 1.0).abs() <= 1.0e-6);
            assert!(
                probabilities[row * 5 + visible..(row + 1) * 5]
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
        assert!(probabilities[2] > probabilities[1]);
        assert_eq!(probabilities[5..10], [0.2; 5]);
    }

    #[test]
    fn selector_forces_sinks_recent_diagonals_and_lower_ties() {
        let context = 140;
        let last_queries = 2;
        let mut probabilities = vec![0.0_f32; last_queries * context];
        probabilities[0 * context + 80] = 1.0;
        probabilities[1 * context + 90] = 1.0;
        let selected =
            vertical_slash_selection(&probabilities, last_queries, context, 31, 101).unwrap();
        assert_eq!(
            selected.vertical_positions[..30],
            (0..30).collect::<Vec<_>>()
        );
        assert!(selected.vertical_positions.contains(&80));
        assert_eq!(
            selected.slash_distances[..100],
            (0..100).collect::<Vec<_>>()
        );
        // Both observations fall inside the forced recent region. The next
        // all-zero tie chooses source diagonal index zero, which maps to the
        // largest distance and proves source-index rather than distance-index tie order.
        assert_eq!(selected.slash_distances[100], 139);
    }

    #[test]
    fn full_selection_control_compacts_in_original_causal_order() {
        let context = 128;
        let probabilities = vec![1.0 / context as f32; context];
        let selected =
            vertical_slash_selection(&probabilities, 1, context, context, context).unwrap();
        assert_eq!(
            selected_positions_for_query(context - 1, &selected).unwrap(),
            (0..context).collect::<Vec<_>>()
        );
    }

    #[test]
    fn selector_rejects_future_probability_mass() {
        let mut probabilities = vec![0.0_f32; 16];
        probabilities[6] = 1.0;
        probabilities[7] = 0.5;
        probabilities[15] = 1.0;
        assert!(vertical_slash_selection(&probabilities, 2, 8, 8, 8).is_err());
    }

    #[test]
    fn minference_last64_fixture_matches_independent_pytorch_selection() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../evals/fixtures/tiny/pw0176-vertical-slash-selector.json"
        ))
        .expect("PW-0176 selector fixture");
        assert_eq!(fixture["schema_version"], 1);
        assert_eq!(
            fixture["source_sha256"],
            "b368e765fcc2021591f7cdf970e4e1a71731ed66f19e97023a13b70a02e6edc2"
        );
        let context = fixture["context"].as_u64().unwrap() as usize;
        let last_queries = fixture["last_queries"].as_u64().unwrap() as usize;
        let query_start = context - last_queries;
        let mut scores = vec![900.0_f32; context * last_queries];
        for row in 0..last_queries {
            let query_position = query_start + row;
            for key in 0..=query_position {
                scores[row * context + key] =
                    ((row * 37 + key * 17) % 29) as f32 / 8.0 - 14.0 / 8.0;
            }
        }
        for spike in fixture["spikes"].as_array().unwrap() {
            let row = spike[0].as_u64().unwrap() as usize;
            let key = spike[1].as_u64().unwrap() as usize;
            scores[row * context + key] = spike[2].as_f64().unwrap() as f32;
        }
        let probabilities = causal_f32_softmax_rows(&scores, last_queries, context).unwrap();
        let selection = vertical_slash_selection(
            &probabilities,
            last_queries,
            context,
            fixture["vertical_size"].as_u64().unwrap() as usize,
            fixture["slash_size"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let expected = |name: &str| {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        };
        assert_eq!(selection.vertical_positions, expected("vertical_positions"));
        assert_eq!(selection.slash_distances, expected("slash_distances"));
        for position in [63, 100, 139] {
            let expected_positions = fixture["selected_positions"][position.to_string()]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(
                selected_positions_for_query(position, &selection).unwrap(),
                expected_positions
            );
        }
        let full =
            vertical_slash_selection(&probabilities, last_queries, context, context, context)
                .unwrap();
        assert_eq!(
            selected_positions_for_query(139, &full).unwrap(),
            expected("full_selection_at_139")
        );
    }
}
