use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Serialize)]
pub struct TensorRecord {
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_bytes: u64,
    pub category: String,
}

#[derive(Debug, Default, Serialize)]
pub struct CategoryTotal {
    pub tensors: u64,
    pub data_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct Census {
    pub schema_version: u32,
    pub tensor_count: usize,
    pub tensor_data_bytes: u64,
    pub shard_file_bytes: u64,
    pub header_and_padding_bytes: u64,
    pub categories: BTreeMap<String, CategoryTotal>,
    pub tensors: Vec<TensorRecord>,
}

fn require_object<'a>(
    value: &'a Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} must be an object"))
}

fn require_u64_pair(value: &Value, context: &str) -> Result<(u64, u64), String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    if values.len() != 2 {
        return Err(format!("{context} must contain two offsets"));
    }
    let start = values[0]
        .as_u64()
        .ok_or_else(|| format!("{context} start is not u64"))?;
    let end = values[1]
        .as_u64()
        .ok_or_else(|| format!("{context} end is not u64"))?;
    if end < start {
        return Err(format!("{context} offsets are reversed"));
    }
    Ok((start, end))
}

pub fn classify_tensor(name: &str) -> &'static str {
    if name.starts_with("visual.") {
        "vision_encoder"
    } else if name.starts_with("audio_encoder.") || name.starts_with("speech_embeddings.") {
        "audio_path"
    } else if name.contains(".mtp.") {
        "mtp"
    } else if name.contains(".mlp.experts.") {
        "routed_experts"
    } else if name.contains(".mlp.gate.weight") {
        "routers"
    } else if name == "model.embed_tokens.weight" {
        "token_embeddings"
    } else if name == "lm_head.weight" {
        "lm_head"
    } else if name.starts_with("model.layers.0.mlp.") {
        "dense_layer_zero"
    } else if name.contains("self_attn")
        || name.contains("layernorm")
        || name == "model.norm.weight"
    {
        "attention_and_norms"
    } else {
        "other_language_or_projector"
    }
}

fn read_safetensors_header(path: &Path) -> Result<(u64, serde_json::Map<String, Value>), String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let file_bytes = file.metadata().map_err(|error| error.to_string())?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES || header_bytes + 8 > file_bytes {
        return Err(format!(
            "{}: invalid header length {header_bytes}",
            path.display()
        ));
    }
    let header_len = usize::try_from(header_bytes).map_err(|_| "header does not fit usize")?;
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let value: Value = serde_json::from_slice(&header)
        .map_err(|error| format!("{}: malformed header JSON: {error}", path.display()))?;
    Ok((
        file_bytes,
        require_object(&value, "safetensors header")?.clone(),
    ))
}

pub fn build_census(index_path: &Path, checkpoint_dir: &Path) -> Result<Census, String> {
    let index: Value = serde_json::from_reader(
        File::open(index_path).map_err(|error| format!("{}: {error}", index_path.display()))?,
    )
    .map_err(|error| format!("{}: {error}", index_path.display()))?;
    let weight_map = require_object(
        index.get("weight_map").ok_or("index lacks weight_map")?,
        "weight_map",
    )?;
    let mut shard_names = BTreeSet::new();
    for shard in weight_map.values() {
        shard_names.insert(
            shard
                .as_str()
                .ok_or("weight_map shard must be a string")?
                .to_owned(),
        );
    }
    // Standalone safetensors not named by the main index are explicit sources too.
    let audio_tokenizer = checkpoint_dir.join("audio_tokenizer/model.safetensors");
    if audio_tokenizer.exists() {
        shard_names.insert("audio_tokenizer/model.safetensors".to_owned());
    }

    let mut records = Vec::new();
    let mut seen_indexed = BTreeSet::new();
    let mut shard_file_bytes = 0_u64;
    for shard in shard_names {
        let path: PathBuf = checkpoint_dir.join(&shard);
        let (file_bytes, header) = read_safetensors_header(&path)?;
        shard_file_bytes = shard_file_bytes
            .checked_add(file_bytes)
            .ok_or("shard byte overflow")?;
        for (name, metadata) in header {
            if name == "__metadata__" {
                continue;
            }
            let object = require_object(&metadata, &name)?;
            let dtype = object
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{name}: missing dtype"))?;
            let shape = object
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name}: missing shape"))?
                .iter()
                .map(|dimension| {
                    dimension
                        .as_u64()
                        .ok_or_else(|| format!("{name}: invalid shape"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (start, end) = require_u64_pair(
                object
                    .get("data_offsets")
                    .ok_or_else(|| format!("{name}: missing offsets"))?,
                &name,
            )?;
            if let Some(expected_shard) = weight_map.get(&name) {
                if expected_shard.as_str() != Some(&shard) {
                    return Err(format!("{name}: index points to a different shard"));
                }
                if !seen_indexed.insert(name.clone()) {
                    return Err(format!("{name}: tensor appears more than once"));
                }
            } else if shard != "audio_tokenizer/model.safetensors" {
                return Err(format!("{name}: tensor is absent from the source index"));
            }
            records.push(TensorRecord {
                category: classify_tensor(&name).to_owned(),
                name,
                shard: shard.clone(),
                dtype: dtype.to_owned(),
                shape,
                data_bytes: end - start,
            });
        }
    }
    if seen_indexed.len() != weight_map.len() {
        return Err(format!(
            "index assigns {} tensors but only {} were found",
            weight_map.len(),
            seen_indexed.len()
        ));
    }
    records.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.shard.cmp(&right.shard))
    });
    let tensor_data_bytes = records.iter().try_fold(0_u64, |total, record| {
        total
            .checked_add(record.data_bytes)
            .ok_or("tensor byte overflow")
    })?;
    let mut categories: BTreeMap<String, CategoryTotal> = BTreeMap::new();
    for record in &records {
        let total = categories.entry(record.category.clone()).or_default();
        total.tensors += 1;
        total.data_bytes += record.data_bytes;
    }
    Ok(Census {
        schema_version: 1,
        tensor_count: records.len(),
        tensor_data_bytes,
        shard_file_bytes,
        header_and_padding_bytes: shard_file_bytes
            .checked_sub(tensor_data_bytes)
            .ok_or("tensor bytes exceed shard bytes")?,
        categories,
        tensors: records,
    })
}

pub fn write_census(census: &Census, output: &Path) -> Result<(), String> {
    let mut file =
        File::create(output).map_err(|error| format!("{}: {error}", output.display()))?;
    serde_json::to_writer_pretty(&mut file, census).map_err(|error| error.to_string())?;
    use std::io::Write;
    file.write_all(b"\n").map_err(|error| error.to_string())
}

pub fn read_header_length(path: &Path) -> Result<u64, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| error.to_string())?;
    Ok(u64::from_le_bytes(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_categories_are_explicit() {
        assert_eq!(
            classify_tensor("model.layers.1.mlp.experts.4.up_proj.weight"),
            "routed_experts"
        );
        assert_eq!(classify_tensor("model.layers.3.mlp.gate.weight"), "routers");
        assert_eq!(
            classify_tensor("visual.blocks.0.attn.qkv.weight"),
            "vision_encoder"
        );
        assert_eq!(
            classify_tensor("audio_encoder.layers.0.weight"),
            "audio_path"
        );
    }
}
