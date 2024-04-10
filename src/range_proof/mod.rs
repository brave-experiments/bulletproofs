#![allow(non_snake_case)]
#![doc = include_str!("../../docs/range-proof-protocol.md")]

extern crate alloc;

use alloc::vec::Vec;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{prelude::thread_rng, Rng};
use ark_std::One;

use core::iter;
use std::marker::PhantomData;

use merlin::Transcript;

use crate::errors::ProofError;
use crate::generators::{BulletproofGens, PedersenGens};
use crate::inner_product_proof::InnerProductProof;
use crate::transcript::TranscriptProtocol;
use crate::util;

// Modules for MPC protocol
pub mod dealer;
pub mod messages;
pub mod party;

/// The `RangeProof` struct represents a proof that one or more values
/// are in a range.
///
/// The `RangeProof` struct contains functions for creating and
/// verifying aggregated range proofs.  The single-value case is
/// implemented as a special case of aggregated range proofs.
///
/// The bitsize of the range, as well as the list of commitments to
/// the values, are not included in the proof, and must be known to
/// the verifier.
///
/// This implementation requires that both the bitsize `n` and the
/// aggregation size `m` be powers of two, so that `n = 8, 16, 32, 64`
/// and `m = 1, 2, 4, 8, 16, ...`.  Note that the aggregation size is
/// not given as an explicit parameter, but is determined by the
/// number of values or commitments passed to the prover or verifier.
///
/// # Note
///
/// For proving, these functions run the multiparty aggregation
/// protocol locally.  That API is exposed in the [`aggregation`](::range_proof_mpc)
/// module and can be used to perform online aggregation between
/// parties without revealing secret values to each other.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RangeProof<C: AffineRepr, F: Field + PrimeField> {
    /// Commitment to the bits of the value
    A: C,
    /// Commitment to the blinding factors
    S: C,
    /// Commitment to the \\(t_1\\) coefficient of \\( t(x) \\)
    T_1: C,
    /// Commitment to the \\(t_2\\) coefficient of \\( t(x) \\)
    T_2: C,
    /// Evaluation of the polynomial \\(t(x)\\) at the challenge point \\(x\\)
    t_x: C::ScalarField,
    /// Blinding factor for the synthetic commitment to \\(t(x)\\)
    t_x_blinding: C::ScalarField,
    /// Blinding factor for the synthetic commitment to the inner-product arguments
    e_blinding: C::ScalarField,
    /// Proof data for the inner-product argument.
    ipp_proof: InnerProductProof<C>,
    _marker_f: PhantomData<F>,
}

impl<C: AffineRepr, F: Field + PrimeField> RangeProof<C, F> {
    /// Create a rangeproof for a given pair of value `v` and
    /// blinding scalar `v_blinding`.
    /// This is a convenience wrapper around [`RangeProof::prove_single_with_rng`],
    /// passing in a threadsafe RNG.
    #[cfg(feature = "std")]
    pub fn prove_single(
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        transcript: &mut Transcript,
        v: u64,
        v_blinding: &C::ScalarField,
        n: usize,
    ) -> Result<(RangeProof<C, F>, C), ProofError> {
        RangeProof::prove_single_with_rng(
            bp_gens,
            pc_gens,
            transcript,
            v,
            v_blinding,
            n,
            &mut thread_rng(),
        )
    }

    /// Create a rangeproof for a given pair of value `v` and
    /// blinding scalar `v_blinding`.
    /// This is a convenience wrapper around [`RangeProof::prove_multiple`].
    /// ```
    pub fn prove_single_with_rng<R: Rng>(
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        transcript: &mut Transcript,
        v: u64,
        v_blinding: &C::ScalarField,
        n: usize,
        rng: &mut R,
    ) -> Result<(RangeProof<C, F>, C), ProofError> {
        let (p, Vs) = RangeProof::prove_multiple_with_rng(
            bp_gens,
            pc_gens,
            transcript,
            &[v],
            &[*v_blinding],
            n,
            rng,
        )?;
        Ok((p, Vs[0]))
    }

    /// Create a rangeproof for a set of values.
    pub fn prove_multiple_with_rng<R: Rng>(
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        transcript: &mut Transcript,
        values: &[u64],
        blindings: &[C::ScalarField],
        n: usize,
        rng: &mut R,
    ) -> Result<(RangeProof<C, F>, Vec<C>), ProofError> {
        use self::dealer::*;
        use self::party::*;

        if values.len() != blindings.len() {
            return Err(ProofError::WrongNumBlindingFactors);
        }

        let dealer = Dealer::new(bp_gens, pc_gens, transcript, n, values.len())?;

        let parties: Vec<_> = values
            .iter()
            .zip(blindings.iter())
            .map(|(&v, &v_blinding)| Party::<C, F>::new(bp_gens, pc_gens, v, v_blinding, n))
            // Collect the iterator of Results into a Result<Vec>, then unwrap it
            .collect::<Result<Vec<_>, _>>()?;

        let (parties, bit_commitments): (Vec<_>, Vec<_>) = parties
            .into_iter()
            .enumerate()
            .map(|(j, p)| {
                p.assign_position_with_rng(j, rng)
                    .expect("We already checked the parameters, so this should never happen")
            })
            .unzip();

        let value_commitments: Vec<_> = bit_commitments.iter().map(|c| c.V_j).collect();

        let (dealer, bit_challenge) = dealer.receive_bit_commitments(bit_commitments)?;

        let (parties, poly_commitments): (Vec<_>, Vec<_>) = parties
            .into_iter()
            .map(|p| p.apply_challenge_with_rng(&bit_challenge, rng))
            .unzip();

        let (dealer, poly_challenge) = dealer.receive_poly_commitments(poly_commitments)?;

        let proof_shares: Vec<_> = parties
            .into_iter()
            .map(|p| p.apply_challenge(&poly_challenge))
            // Collect the iterator of Results into a Result<Vec>, then unwrap it
            .collect::<Result<Vec<_>, _>>()?;

        let proof = dealer.receive_trusted_shares(&proof_shares)?;

        Ok((proof, value_commitments))
    }

    /// Create a rangeproof for a set of values.
    /// This is a convenience wrapper around [`RangeProof::prove_multiple_with_rng`],
    /// passing in a threadsafe RNG.
    #[cfg(feature = "std")]
    pub fn prove_multiple(
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        transcript: &mut Transcript,
        values: &[u64],
        blindings: &[C::ScalarField],
        n: usize,
    ) -> Result<(RangeProof<C, F>, Vec<C>), ProofError> {
        RangeProof::prove_multiple_with_rng(
            bp_gens,
            pc_gens,
            transcript,
            values,
            blindings,
            n,
            &mut thread_rng(),
        )
    }

    /// Verifies a rangeproof for a given value commitment \\(V\\).
    ///
    /// This is a convenience wrapper around [`RangeProof::verify_single_with_rng`],
    /// passing in a threadsafe RNG.
    //#[cfg(feature = "std")]
    pub fn verify_single(
        &self,
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        transcript: &mut Transcript,
        V: &C,
        n: usize,
    ) -> Result<(), ProofError> {
        self.verify_single_with_rng(bp_gens, pc_gens, transcript, V, n, &mut thread_rng())
    }

    /// Verifies a rangeproof for a given value commitment \\(V\\).
    ///
    /// This is a convenience wrapper around `verify_multiple` for the `m=1` case.
    pub fn verify_single_with_rng<R: Rng>(
        &self,
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        transcript: &mut Transcript,
        V: &C,
        n: usize,
        rng: &mut R,
    ) -> Result<(), ProofError> {
        self.verify_multiple_with_rng(bp_gens, pc_gens, transcript, &[*V], n, rng)
    }

    /// Verifies an aggregated rangeproof for the given value commitments.
    pub fn verify_multiple_with_rng<R: Rng>(
        &self,
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        transcript: &mut Transcript,
        value_commitments: &[C],
        n: usize,
        rng: &mut R,
    ) -> Result<(), ProofError> {
        let m = value_commitments.len();

        // First, replay the "interactive" protocol using the proof
        // data to recompute all challenges.
        if !(n == 8 || n == 16 || n == 32 || n == 64) {
            return Err(ProofError::InvalidBitsize);
        }
        if bp_gens.gens_capacity < n {
            return Err(ProofError::InvalidGeneratorsLength);
        }
        if bp_gens.party_capacity < m {
            return Err(ProofError::InvalidGeneratorsLength);
        }

        transcript.rangeproof_domain_sep(n as u64, m as u64);

        for V in value_commitments.iter() {
            // Allow the commitments to be zero (0 value, 0 blinding)
            // See https://github.com/dalek-cryptography/bulletproofs/pull/248#discussion_r255167177
            transcript.append_point(b"V", V);
        }

        transcript.validate_and_append_point(b"A", &self.A)?;
        transcript.validate_and_append_point(b"S", &self.S)?;

        let y = transcript.challenge_scalar::<C>(b"y");
        let z = transcript.challenge_scalar::<C>(b"z");
        let zz = z * z;
        let minus_z = -z;

        transcript.validate_and_append_point(b"T_1", &self.T_1)?;
        transcript.validate_and_append_point(b"T_2", &self.T_2)?;

        let x = transcript.challenge_scalar::<C>(b"x");

        transcript.append_scalar::<C>(b"t_x", &self.t_x);
        transcript.append_scalar::<C>(b"t_x_blinding", &self.t_x_blinding);
        transcript.append_scalar::<C>(b"e_blinding", &self.e_blinding);

        let w = transcript.challenge_scalar::<C>(b"w");

        // Challenge value for batching statements to be verified
        let c = C::ScalarField::rand(rng);

        let (x_sq, x_inv_sq, s) = self.ipp_proof.verification_scalars(n * m, transcript)?;
        let s_inv = s.iter().rev();

        let a = self.ipp_proof.a;
        let b = self.ipp_proof.b;

        // Construct concat_z_and_2, an iterator of the values of
        // z^0 * \vec(2)^n || z^1 * \vec(2)^n || ... || z^(m-1) * \vec(2)^n
        let powers_of_2: Vec<C::ScalarField> =
            util::exp_iter(C::ScalarField::from(2u64)).take(n).collect();
        let concat_z_and_2: Vec<C::ScalarField> = util::exp_iter(z)
            .take(m)
            .flat_map(|exp_z| powers_of_2.iter().map(move |exp_2| (*exp_2) * exp_z))
            .collect();

        let g = s.iter().map(|s_i| minus_z - a * s_i);
        let h = s_inv
            .zip(util::exp_iter(y.inverse().unwrap()))
            .zip(concat_z_and_2.iter())
            .map(|((s_i_inv, exp_y_inv), z_and_2)| z + exp_y_inv * (zz * z_and_2 - b * s_i_inv));

        let value_commitment_scalars = util::exp_iter(z).take(m).map(|z_exp| c * zz * z_exp);
        let basepoint_scalar = w * (self.t_x - a * b) + c * (delta(n, m, &y, &z) - self.t_x);

        let mega_check = C::Group::msm(
            iter::once(self.A)
                .chain(iter::once(self.S))
                .chain(iter::once(self.T_1))
                .chain(iter::once(self.T_2))
                .chain(self.ipp_proof.L_vec.iter().map(|L| L.clone()))
                .chain(self.ipp_proof.R_vec.iter().map(|R| R.clone()))
                .chain(iter::once(Some(pc_gens.B_blinding).unwrap()))
                .chain(iter::once(Some(pc_gens.B).unwrap()))
                .chain(bp_gens.G(n, m).map(|&x| Some(x).unwrap()))
                .chain(bp_gens.H(n, m).map(|&x| Some(x).unwrap()))
                .chain(value_commitments.iter().map(|V| V.clone()))
                .collect::<Vec<C>>()
                .as_slice(),
            iter::once(C::ScalarField::one())
                .chain(iter::once(x))
                .chain(iter::once(c * x))
                .chain(iter::once(c * x * x))
                .chain(x_sq.iter().cloned())
                .chain(x_inv_sq.iter().cloned())
                .chain(iter::once(-self.e_blinding - c * self.t_x_blinding))
                .chain(iter::once(basepoint_scalar))
                .chain(g)
                .chain(h)
                .chain(value_commitment_scalars)
                .collect::<Vec<C::ScalarField>>()
                .as_slice(),
        )
        .map_err(|_| ProofError::InvalidInputLength)?;

        if mega_check.into_affine().is_zero() {
            Ok(())
        } else {
            Err(ProofError::VerificationError)
        }
    }

    /// Verifies an aggregated rangeproof for the given value commitments.
    /// This is a convenience wrapper around [`RangeProof::verify_multiple_with_rng`],
    /// passing in a threadsafe RNG.
    // Currently not needed in rewards proofs
    #[cfg(feature = "std")]
    pub fn verify_multiple(
        &self,
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        transcript: &mut Transcript,
        value_commitments: &[C],
        n: usize,
    ) -> Result<(), ProofError> {
        self.verify_multiple_with_rng(
            bp_gens,
            pc_gens,
            transcript,
            value_commitments,
            n,
            &mut thread_rng(),
        )
    }
}

/// Compute
/// \\[
/// \delta(y,z) = (z - z^{2}) \langle \mathbf{1}, {\mathbf{y}}^{n \cdot m} \rangle - \sum_{j=0}^{m-1} z^{j+3} \cdot \langle \mathbf{1}, {\mathbf{2}}^{n \cdot m} \rangle
/// \\]
fn delta<S: Field>(n: usize, m: usize, y: &S, z: &S) -> S {
    let sum_y = util::sum_of_powers(y, n * m);
    let sum_2 = util::sum_of_powers(&S::from(2u64), n);
    let sum_z = util::sum_of_powers(z, m);

    ((*z) - (*z) * z) * sum_y - (*z) * z * z * sum_2 * sum_z
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_ff::{One, Zero};

    use ark_ec::short_weierstrass::Affine;
    use ark_secp256k1::{Config as SecpConfig, Fq as SecpBaseField};

    type Scalar = <Affine<SecpConfig> as AffineRepr>::ScalarField;

    #[test]
    fn test_delta() {
        let mut rng = rand::thread_rng();
        let y = Scalar::rand(&mut rng);
        let z = Scalar::rand(&mut rng);

        // Choose n = 256 to ensure we overflow the group order during
        // the computation, to check that that's done correctly
        let n = 256;

        // code copied from previous implementation
        let z2 = z * z;
        let z3 = z2 * z;
        let mut power_g = Scalar::zero();
        let mut exp_y = Scalar::one(); // start at y^0 = 1
        let mut exp_2 = Scalar::one(); // start at 2^0 = 1
        for _ in 0..n {
            power_g += (z - z2) * exp_y - z3 * exp_2;

            exp_y = exp_y * y; // y^i -> y^(i+1)
            exp_2 = exp_2 + exp_2; // 2^i -> 2^(i+1)
        }

        assert_eq!(power_g, delta(n, 1, &y, &z),);
    }

    /// Given a bitsize `n`, test the following:
    ///
    /// 1. Generate `m` random values and create a proof they are all in range;
    /// 2. Serialize to wire format;
    /// 3. Deserialize from wire format;
    /// 4. Verify the proof.
    fn singleparty_create_and_verify_helper<C, F>(n: usize, m: usize)
    where
        C: AffineRepr,
        F: Field + PrimeField,
    {
        // Split the test into two scopes, so that it's explicit what
        // data is shared between the prover and the verifier.

        // Both prover and verifier have access to the generators and the proof
        let max_bitsize = 64;
        let max_parties = 8;
        let pc_gens = PedersenGens::default();
        let bp_gens = BulletproofGens::new(max_bitsize, max_parties);

        // Prover's scope
        let (proof_bytes, value_commitments) = {
            let mut rng = rand::thread_rng();

            // 0. Create witness data
            let (min, max) = (0u64, ((1u128 << n) - 1) as u64);
            let values: Vec<u64> = (0..m).map(|_| rng.gen_range(min..max)).collect();
            let blindings: Vec<C::ScalarField> =
                (0..m).map(|_| C::ScalarField::rand(&mut rng)).collect();

            // 1. Create the proof
            let mut transcript = Transcript::new(b"AggregatedRangeProofTest");
            let (proof, value_commitments) = RangeProof::<C, F>::prove_multiple(
                &bp_gens,
                &pc_gens,
                &mut transcript,
                &values,
                &blindings,
                n,
            )
            .unwrap();

            // Serialize proof
            let mut rp = Vec::new();
            proof.serialize_compressed(&mut rp).unwrap();

            // 2. Return serialized proof and value commitments
            (rp, value_commitments)
        };

        // Verifier's scope
        {
            // 3. Deserialize
            let proof: RangeProof<C, F> =
                RangeProof::deserialize_compressed(&*proof_bytes).unwrap();

            // 4. Verify with the same customization label as above
            let mut transcript = Transcript::new(b"AggregatedRangeProofTest");

            assert!(proof
                .verify_multiple(&bp_gens, &pc_gens, &mut transcript, &value_commitments, n)
                .is_ok());
        }
    }

    #[test]
    fn create_and_verify_n_32_m_1() {
        singleparty_create_and_verify_helper::<Affine<SecpConfig>, SecpBaseField>(32, 1);
    }

    #[test]
    fn create_and_verify_n_32_m_2() {
        singleparty_create_and_verify_helper::<Affine<SecpConfig>, SecpBaseField>(32, 2);
    }

    #[test]
    fn create_and_verify_n_32_m_4() {
        singleparty_create_and_verify_helper::<Affine<SecpConfig>, SecpBaseField>(32, 4);
    }

    #[test]
    fn create_and_verify_n_32_m_8() {
        singleparty_create_and_verify_helper::<Affine<SecpConfig>, SecpBaseField>(32, 8);
    }

    #[test]
    fn create_and_verify_n_64_m_1() {
        singleparty_create_and_verify_helper::<Affine<SecpConfig>, SecpBaseField>(64, 1);
    }

    #[test]
    fn create_and_verify_n_64_m_2() {
        singleparty_create_and_verify_helper::<Affine<SecpConfig>, SecpBaseField>(64, 2);
    }

    #[test]
    fn create_and_verify_n_64_m_4() {
        singleparty_create_and_verify_helper::<Affine<SecpConfig>, SecpBaseField>(64, 4);
    }

    #[test]
    fn create_and_verify_n_64_m_8() {
        singleparty_create_and_verify_helper::<Affine<SecpConfig>, SecpBaseField>(64, 8);
    }

    #[test]
    fn detect_dishonest_party_during_aggregation() {
        use self::dealer::*;
        use self::party::*;

        use crate::errors::MPCError;

        // Simulate four parties, two of which will be dishonest and use a 64-bit value.
        let m = 4;
        let n = 32;

        let pc_gens = PedersenGens::<Affine<SecpConfig>>::default();
        let bp_gens = BulletproofGens::new(n, m);

        let mut rng = rand::thread_rng();
        let mut transcript = Transcript::new(b"AggregatedRangeProofTest");

        // Parties 0, 2 are honest and use a 32-bit value
        let v0 = rng.gen::<u32>() as u64;
        let v0_blinding = Scalar::rand(&mut rng);
        let party0 =
            Party::<Affine<SecpConfig>, SecpBaseField>::new(&bp_gens, &pc_gens, v0, v0_blinding, n)
                .unwrap();

        let v2 = rng.gen::<u32>() as u64;
        let v2_blinding = Scalar::rand(&mut rng);
        let party2 =
            Party::<Affine<SecpConfig>, SecpBaseField>::new(&bp_gens, &pc_gens, v2, v2_blinding, n)
                .unwrap();

        // Parties 1, 3 are dishonest and use a 64-bit value
        let v1 = rng.gen::<u64>();
        let v1_blinding = Scalar::rand(&mut rng);
        let party1 =
            Party::<Affine<SecpConfig>, SecpBaseField>::new(&bp_gens, &pc_gens, v1, v1_blinding, n)
                .unwrap();

        let v3 = rng.gen::<u64>();
        let v3_blinding = Scalar::rand(&mut rng);
        let party3 =
            Party::<Affine<SecpConfig>, SecpBaseField>::new(&bp_gens, &pc_gens, v3, v3_blinding, n)
                .unwrap();

        let dealer = Dealer::<Affine<SecpConfig>, SecpBaseField>::new(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            n,
            m,
        )
        .unwrap();

        let (party0, bit_com0) = party0.assign_position(0).unwrap();
        let (party1, bit_com1) = party1.assign_position(1).unwrap();
        let (party2, bit_com2) = party2.assign_position(2).unwrap();
        let (party3, bit_com3) = party3.assign_position(3).unwrap();

        let (dealer, bit_challenge) = dealer
            .receive_bit_commitments(vec![bit_com0, bit_com1, bit_com2, bit_com3])
            .unwrap();

        let (party0, poly_com0) = party0.apply_challenge(&bit_challenge);
        let (party1, poly_com1) = party1.apply_challenge(&bit_challenge);
        let (party2, poly_com2) = party2.apply_challenge(&bit_challenge);
        let (party3, poly_com3) = party3.apply_challenge(&bit_challenge);

        let (dealer, poly_challenge) = dealer
            .receive_poly_commitments(vec![poly_com0, poly_com1, poly_com2, poly_com3])
            .unwrap();

        let share0 = party0.apply_challenge(&poly_challenge).unwrap();
        let share1 = party1.apply_challenge(&poly_challenge).unwrap();
        let share2 = party2.apply_challenge(&poly_challenge).unwrap();
        let share3 = party3.apply_challenge(&poly_challenge).unwrap();

        match dealer.receive_shares(&[share0, share1, share2, share3]) {
            Err(MPCError::MalformedProofShares { bad_shares }) => {
                assert_eq!(bad_shares, vec![1, 3]);
            }
            Err(_) => {
                panic!("Got wrong error type from malformed shares");
            }
            Ok(_) => {
                panic!("The proof was malformed, but it was not detected");
            }
        }
    }

    #[test]
    fn detect_dishonest_dealer_during_aggregation() {
        use self::dealer::*;
        use self::party::*;
        use crate::errors::MPCError;

        // Simulate one party
        let m = 1;
        let n = 32;

        let pc_gens = PedersenGens::<Affine<SecpConfig>>::default();
        let bp_gens = BulletproofGens::new(n, m);

        let mut rng = rand::thread_rng();
        let mut transcript = Transcript::new(b"AggregatedRangeProofTest");

        let v0 = rng.gen::<u32>() as u64;
        let v0_blinding = Scalar::rand(&mut rng);
        let party0 =
            Party::<Affine<SecpConfig>, SecpBaseField>::new(&bp_gens, &pc_gens, v0, v0_blinding, n)
                .unwrap();

        let dealer = Dealer::<Affine<SecpConfig>, SecpBaseField>::new(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            n,
            m,
        )
        .unwrap();

        // Now do the protocol flow as normal....

        let (party0, bit_com0) = party0.assign_position(0).unwrap();

        let (dealer, bit_challenge) = dealer.receive_bit_commitments(vec![bit_com0]).unwrap();

        let (party0, poly_com0) = party0.apply_challenge(&bit_challenge);

        let (_dealer, mut poly_challenge) =
            dealer.receive_poly_commitments(vec![poly_com0]).unwrap();

        // But now simulate a malicious dealer choosing x = 0
        poly_challenge.x = Scalar::zero();

        let maybe_share0 = party0.apply_challenge(&poly_challenge);

        assert!(maybe_share0.unwrap_err() == MPCError::MaliciousDealer);
    }
}
