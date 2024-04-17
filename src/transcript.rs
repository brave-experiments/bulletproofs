//! Defines a `TranscriptProtocol` trait for using a Merlin transcript.

use ark_ec::AffineRepr;
use ark_ff::Field;
use merlin::Transcript;

use crate::errors::ProofError;
use crate::util;

pub trait TranscriptProtocol {
    /// Append a domain separator for an `n`-bit, `m`-party range proof.
    fn rangeproof_domain_sep(&mut self, n: u64, m: u64);

    /// Append a domain separator for a length-`n` inner product proof.
    fn innerproduct_domain_sep(&mut self, n: u64);

    /// Append a domain separator for a constraint system.
    fn r1cs_domain_sep(&mut self);

    /// Commit a domain separator for a CS without randomized constraints.
    fn r1cs_1phase_domain_sep(&mut self);

    /// Commit a domain separator for a CS with randomized constraints.
    fn r1cs_2phase_domain_sep(&mut self);

    /// Append a `scalar` with the given `label`.
    fn append_scalar<C: AffineRepr>(&mut self, label: &'static [u8], scalar: &C::ScalarField);

    /// Append a `point` with the given `label`.
    fn append_point<C: AffineRepr>(&mut self, label: &'static [u8], point: &C);

    /// Check that a point is not the identity, then append it to the
    /// transcript.  Otherwise, return an error.
    fn validate_and_append_point<C: AffineRepr>(
        &mut self,
        label: &'static [u8],
        point: &C,
    ) -> Result<(), ProofError>;

    /// Compute a `label`ed challenge variable.
    fn challenge_scalar<C: AffineRepr>(&mut self, label: &'static [u8]) -> C::ScalarField;
}

impl TranscriptProtocol for Transcript {
    fn rangeproof_domain_sep(&mut self, n: u64, m: u64) {
        self.append_message(b"dom-sep", b"rangeproof v1");
        self.append_u64(b"n", n);
        self.append_u64(b"m", m);
    }

    fn innerproduct_domain_sep(&mut self, n: u64) {
        self.append_message(b"dom-sep", b"ipp v1");
        self.append_u64(b"n", n);
    }

    fn r1cs_domain_sep(&mut self) {
        self.append_message(b"dom-sep", b"r1cs v1");
    }

    fn r1cs_1phase_domain_sep(&mut self) {
        self.append_message(b"dom-sep", b"r1cs-1phase");
    }

    fn r1cs_2phase_domain_sep(&mut self) {
        self.append_message(b"dom-sep", b"r1cs-2phase");
    }

    fn append_scalar<C: AffineRepr>(&mut self, label: &'static [u8], scalar: &C::ScalarField) {
        self.append_message(label, &util::field_as_bytes(scalar));
    }

    fn append_point<C: AffineRepr>(&mut self, label: &'static [u8], point: &C) {
        let mut bytes = Vec::new();
        if let Err(e) = point.serialize_compressed(&mut bytes) {
            panic!("{}", e)
        }
        self.append_message(label, &bytes);
    }

    fn validate_and_append_point<C: AffineRepr>(
        &mut self,
        label: &'static [u8],
        point: &C,
    ) -> Result<(), ProofError> {
        if point.is_zero() {
            Err(ProofError::VerificationError)
        } else {
            let mut bytes = Vec::new();
            if let Err(e) = point.serialize_compressed(&mut bytes) {
                panic!("{}", e)
            }
            self.append_message(label, &bytes);
            Ok(())
        }
    }

    fn challenge_scalar<C: AffineRepr>(&mut self, label: &'static [u8]) -> C::ScalarField {
        extern crate crypto;
        use crypto::digest::Digest;
        use crypto::sha3::Sha3;

        let mut bytes = [0u8; 64];
        self.challenge_bytes(label, &mut bytes);

        for i in 0..=u8::max_value() {
            let mut sha = Sha3::sha3_256();
            sha.input(&bytes);
            sha.input(&[i]);
            let mut buf = [0u8; 32];

            sha.result(&mut buf);
            let res = <C::ScalarField as Field>::from_random_bytes(&buf);

            if let Some(scalar) = res {
                return scalar;
            }
        }
        panic!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::AffineRepr;
    use ark_ff::UniformRand;
    use ark_secp256r1::Affine;
    use ark_serialize::CanonicalSerialize;
    use crypto::{digest::Digest, sha3::Sha3};
    use strobe_rs::{SecParam, Strobe};

    type Scalar = <Affine as AffineRepr>::ScalarField;

    /// Create a TestTranscript as a full strobe implementation to compare
    /// the operations against it
    struct TestTranscript {
        state: Strobe,
    }

    impl TestTranscript {
        /// Strobe init; meta-AD(label)
        pub fn new(label: &[u8]) -> TestTranscript {
            let merlin_protocol_label: &[u8] = b"Merlin v1.0";
            let mut tt = TestTranscript {
                state: Strobe::new(merlin_protocol_label, SecParam::B128),
            };
            tt.append_message(b"dom-sep", label);

            tt
        }

        /// Strobe op: meta-AD(label || len(message)); AD(message)
        pub fn append_message(&mut self, label: &[u8], message: &[u8]) {
            // metadata = label || len(message);
            let mut metadata: Vec<u8> = Vec::with_capacity(label.len() + 4);
            metadata.extend_from_slice(label);
            metadata.extend_from_slice(&Self::encode_usize_as_u32(message.len()));

            self.state.meta_ad(&metadata, false);
            self.state.ad(message, false);
        }

        /// Strobe op: meta-AD(label || len(dest)); PRF into challenge_bytes
        pub fn challenge_bytes(&mut self, label: &[u8], dest: &mut [u8]) {
            let prf_len = dest.len();

            // metadata = label || len(challenge_bytes)
            let mut metadata: Vec<u8> = Vec::with_capacity(label.len() + 4);
            metadata.extend_from_slice(label);
            metadata.extend_from_slice(&Self::encode_usize_as_u32(prf_len));

            self.state.meta_ad(&metadata, false);
            self.state.prf(dest, false);
        }

        /// Serializes a point and appends it to the transcript
        pub fn append_point<C: AffineRepr>(&mut self, label: &[u8], point: &C) {
            let mut bytes: Vec<u8> = Vec::new();
            point.serialize_compressed(&mut bytes).unwrap();
            self.append_message(label, &bytes);
        }

        /// Validates a point, then serializes and appends it to the transcript
        pub fn validate_and_append_point<C: AffineRepr>(
            &mut self,
            label: &[u8],
            point: &C,
        ) -> Result<(), ProofError> {
            if point.is_zero() {
                Err(ProofError::VerificationError)
            } else {
                self.append_point(label, point);
                Ok(())
            }
        }

        /// Serializes a scalar and appends it to the transcript
        pub fn append_scalar<C: AffineRepr>(
            &mut self,
            label: &'static [u8],
            scalar: &C::ScalarField,
        ) {
            let mut bytes: Vec<u8> = Vec::new();
            scalar.serialize_compressed(&mut bytes).unwrap();
            self.append_message(label, &bytes);
        }

        /// Challenge bytes and reconstruct scalar from bytes
        pub fn challenge_scalar<C: AffineRepr>(&mut self, label: &'static [u8]) -> C::ScalarField {
            let mut bytes = [0u8; 64];
            self.challenge_bytes(label, &mut bytes);

            // reconstruct scalar from bytes
            for i in 0..=u8::max_value() {
                let mut sha = Sha3::sha3_256();
                sha.input(&bytes);
                sha.input(&[i]);

                let mut buf = [0u8; 32];
                sha.result(&mut buf);

                if let Some(scalar) = <C::ScalarField>::from_random_bytes(&bytes) {
                    return scalar;
                }
            }
            panic!()
        }

        /// Appends the domain seperation for range proofs
        pub fn rangeproof_domain_sep(&mut self, n: u64, m: u64) {
            self.append_message(b"dom-sep", b"rangeproof v1");
            self.append_message(b"n", &Self::encode_u64(n));
            self.append_message(b"m", &Self::encode_u64(m));
        }

        /// Helper function to encode usize as u32 in little endian byte order
        fn encode_usize_as_u32(x: usize) -> [u8; 4] {
            use byteorder::{ByteOrder, LittleEndian};

            assert!(x <= (u32::max_value() as usize));

            let mut buf = [0; 4];
            LittleEndian::write_u32(&mut buf, x as u32);
            buf
        }

        /// Helper function to encode u64 in little endian byte order
        fn encode_u64(x: u64) -> [u8; 8] {
            use byteorder::{ByteOrder, LittleEndian};

            let mut buf = [0; 8];
            LittleEndian::write_u64(&mut buf, x);
            buf
        }
    }

    #[test]
    fn test_rangeproof_domain_sep() {
        let mut real_transcript = Transcript::new(b"test protocol");
        let mut test_transcript = TestTranscript::new(b"test protocol");

        real_transcript.rangeproof_domain_sep(64, 1);
        test_transcript.rangeproof_domain_sep(64, 1);

        let mut real_challenge = [0u8; 32];
        let mut test_challenge = [0u8; 32];

        real_transcript.challenge_bytes(b"challenge", &mut real_challenge);
        test_transcript.challenge_bytes(b"challenge", &mut test_challenge);

        assert_eq!(real_challenge, test_challenge);
    }

    /// This test creates two transcripts and appends a random point to both.
    /// We then compare the result of both transcripts when challenging
    /// the output.
    #[test]
    fn test_append_point() {
        let mut real_transcript = Transcript::new(b"test protocol");
        let mut test_transcript = TestTranscript::new(b"test protocol");

        let mut rng = ark_std::test_rng();
        let point = Affine::rand(&mut rng);

        real_transcript.append_point(b"test label", &point);
        test_transcript.append_point(b"test label", &point);

        let mut real_challenge = [0u8; 32];
        let mut test_challenge = [0u8; 32];

        real_transcript.challenge_bytes(b"challenge", &mut real_challenge);
        test_transcript.challenge_bytes(b"challenge", &mut test_challenge);

        assert_eq!(real_challenge, test_challenge);
    }

    #[test]
    fn test_validate_and_append_point() {
        let mut real_transcript = Transcript::new(b"test protocol");
        let mut test_transcript = TestTranscript::new(b"test protocol");

        let mut rng = ark_std::test_rng();
        let point = Affine::rand(&mut rng);

        real_transcript
            .validate_and_append_point(b"test label", &point)
            .unwrap();
        test_transcript
            .validate_and_append_point(b"test label", &point)
            .unwrap();

        let mut real_challenge = [0u8; 32];
        let mut test_challenge = [0u8; 32];

        real_transcript.challenge_bytes(b"challenge", &mut real_challenge);
        test_transcript.challenge_bytes(b"challenge", &mut test_challenge);

        assert_eq!(real_challenge, test_challenge);

        // test validation failing
        assert_eq!(
            real_transcript.validate_and_append_point(b"test label", &<Affine>::zero()),
            Err(ProofError::VerificationError)
        );
        assert_eq!(
            test_transcript.validate_and_append_point(b"test label", &<Affine>::zero()),
            Err(ProofError::VerificationError)
        );
    }

    #[test]
    fn test_challenge_scalar() {
        let mut real_transcript = Transcript::new(b"test protocol");
        let mut test_transcript = TestTranscript::new(b"test protocol");

        let real_scalar: Scalar = real_transcript.challenge_scalar::<Affine>(b"challenge");
        let test_scalar: Scalar = test_transcript.challenge_scalar::<Affine>(b"challenge");

        assert_eq!(real_scalar, test_scalar);
    }

    #[test]
    fn test_append_scalar() {
        let mut real_transcript = Transcript::new(b"test protocol");
        let mut test_transcript = TestTranscript::new(b"test protocol");

        let mut rng = ark_std::test_rng();
        let random_scalar: Scalar = Scalar::rand(&mut rng);

        real_transcript.append_scalar::<Affine>(b"test label", &random_scalar);
        test_transcript.append_scalar::<Affine>(b"test label", &random_scalar);

        let mut real_challenge = [0u8; 32];
        let mut test_challenge = [0u8; 32];

        real_transcript.challenge_bytes(b"challenge", &mut real_challenge);
        test_transcript.challenge_bytes(b"challenge", &mut test_challenge);

        assert_eq!(real_challenge, test_challenge);
    }
}
