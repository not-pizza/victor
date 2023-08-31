use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PackedVector {
    data: Vec<u8>,
    min: f32,
    max: f32,
}

impl PackedVector {
    fn pack(vector: &[f32]) -> Self {
        let min = vector.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let data = vector
            .iter()
            .map(|&value| {
                let normalized = (value - min) / (max - min);
                // Clamping is necessary to ensure floating point inaccuracies don't result in values <0 or >255
                (normalized * 255.0).round().clamp(0.0, 255.0) as u8
            })
            .collect();

        PackedVector { data, min, max }
    }

    fn unpack(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&bin_index| {
                let normalized = bin_index as f32 / 255.0;
                self.min + normalized * (self.max - self.min)
            })
            .collect()
    }

    pub(crate) fn serialize_embedding<S>(
        #[allow(clippy::ptr_arg)] embedding: &Vec<f32>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let packed = PackedVector::pack(embedding);
        packed.serialize(serializer)
    }

    pub(crate) fn deserialize_embedding<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let packed = PackedVector::deserialize(deserializer)?;
        Ok(packed.unpack())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{
        distributions::{Distribution, Uniform},
        rngs::StdRng,
        SeedableRng,
    };

    #[test]
    fn test_serialization_size() {
        let packed_vector = PackedVector {
            data: vec![0; 1024],
            min: -0.1,
            max: 0.1,
        };
        let unpacked_vector: Vec<f32> = vec![0.0; 1024];
        let bits = bincode::serialize(&packed_vector).unwrap();
        let unpacked_bits = bincode::serialize(&unpacked_vector).unwrap();
        assert_eq!(bits.len(), 1040);
        assert_eq!(unpacked_bits.len(), 4104);
    }

    #[test]
    fn round_trip_zeros() {
        let vector = vec![0.0; 1024];
        let packed_vector = PackedVector::pack(&vector);
        let unpacked_vector = packed_vector.unpack();
        assert_eq!(vector, unpacked_vector);
    }

    #[test]
    fn round_trip_ones() {
        let vector = vec![1.0; 1024];
        let packed_vector = PackedVector::pack(&vector);
        let unpacked_vector = packed_vector.unpack();
        assert_eq!(vector, unpacked_vector);
    }

    #[test]
    fn round_trip_ones_and_zeros() {
        // alternating vector of 1s and 0s, 1024 elements long
        let vector = (0..1024).map(|i| (i % 2) as f32).collect::<Vec<_>>();
        let packed_vector = PackedVector::pack(&vector);
        let unpacked_vector = packed_vector.unpack();
        assert_eq!(vector, unpacked_vector);
    }

    #[test]
    fn test_packed_vector_accuracy() {
        // Fixed seed for deterministic results
        let seed = [0; 32];
        let mut rng = StdRng::from_seed(seed);

        // 1. Generate a vector of 1024 random numbers between -1000 and 1000
        let distribution = Uniform::from(-1000.0f32..=1000.0f32);
        let numbers: Vec<f32> = (0..1024).map(|_| distribution.sample(&mut rng)).collect();

        // 2. Normalize the vector
        let magnitude = numbers.iter().map(|&num| num * num).sum::<f32>().sqrt();
        let normalized: Vec<f32> = numbers.iter().map(|&num| num / magnitude).collect();

        // 3. Pack it
        let packed = PackedVector::pack(&normalized);

        // 4. Unpack it
        let unpacked = packed.unpack();

        // 5. Measure the max loss in accuracy
        let max_loss = normalized
            .iter()
            .zip(unpacked.iter())
            .map(|(original, unpacked)| (original - unpacked).abs())
            .fold(f32::NEG_INFINITY, f32::max);

        // 6. Measure the average change
        let total_loss: f32 = normalized
            .iter()
            .zip(unpacked.iter())
            .map(|(original, unpacked)| (original - unpacked).abs())
            .sum();
        let avg_loss = total_loss / normalized.len() as f32;

        // For a vector database, a small loss in accuracy is acceptable in exchange for a big
        // increase in storage efficiency. We'll set an arbitrary threshold here, say 0.0005.
        assert!(max_loss < 0.0005, "max_loss: {}", max_loss);
        assert!(avg_loss < 0.0002, "avg_loss: {}", avg_loss);

        // Test that packing the unpacked vector again and unpacking it doesn't change it
        let repacked = PackedVector::pack(&unpacked);
        let repacked_unpacked = repacked.unpack();
        assert_eq!(unpacked, repacked_unpacked);
    }
}
