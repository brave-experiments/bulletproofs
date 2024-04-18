//! The `messages` module contains the API for the messages passed between the parties and the dealer
//! in an aggregated multiparty computation protocol.
//!
//! For more explanation of how the `dealer`, `party`, and `messages` modules orchestrate the protocol execution, see
//! [the API for the aggregated multiparty computation protocol](../aggregation/index.html#api-for-the-aggregated-multiparty-computation-protocol).

extern crate alloc;

use alloc::vec::Vec;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::One;
use core::iter;

use crate::generators::{BulletproofGens, PedersenGens};

/// A commitment to the bits of a party's value.
#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BitCommitment<C: AffineRepr> {
    pub(super) V_j: C,
    pub(super) A_j: C,
    pub(super) S_j: C,
}

/// Challenge values derived from all parties' [`BitCommitment`]s.
#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BitChallenge<C: AffineRepr> {
    pub(super) y: C::ScalarField,
    pub(super) z: C::ScalarField,
}

/// A commitment to a party's polynomial coefficents.
#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PolyCommitment<C: AffineRepr> {
    pub(super) T_1_j: C,
    pub(super) T_2_j: C,
}

/// Challenge values derived from all parties' [`PolyCommitment`]s.
#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PolyChallenge<C: AffineRepr> {
    pub(super) x: C::ScalarField,
}

/// A party's proof share, ready for aggregation into the final
/// [`RangeProof`](::RangeProof).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProofShare<C: AffineRepr> {
    pub(super) t_x: C::ScalarField,
    pub(super) t_x_blinding: C::ScalarField,
    pub(super) e_blinding: C::ScalarField,
    pub(super) l_vec: Vec<C::ScalarField>,
    pub(super) r_vec: Vec<C::ScalarField>,
}

impl<C: AffineRepr> ProofShare<C> {
    /// Checks consistency of all sizes in the proof share and returns the size of the l/r vector.
    pub(super) fn check_size(
        &self,
        expected_n: usize,
        bp_gens: &BulletproofGens<C>,
        j: usize,
    ) -> Result<(), ()> {
        if self.l_vec.len() != expected_n {
            return Err(());
        }

        if self.r_vec.len() != expected_n {
            return Err(());
        }

        if expected_n > bp_gens.gens_capacity {
            return Err(());
        }

        if j >= bp_gens.party_capacity {
            return Err(());
        }

        Ok(())
    }

    /// Audit an individual proof share to determine whether it is
    /// malformed.
    pub(super) fn audit_share(
        &self,
        bp_gens: &BulletproofGens<C>,
        pc_gens: &PedersenGens<C>,
        j: usize,
        bit_commitment: &BitCommitment<C>,
        bit_challenge: &BitChallenge<C>,
        poly_commitment: &PolyCommitment<C>,
        poly_challenge: &PolyChallenge<C>,
    ) -> Result<(), ()> {
        use crate::inner_product_proof::inner_product;
        use crate::util;

        let n = self.l_vec.len();

        self.check_size(n, bp_gens, j)?;

        let (y, z) = (&bit_challenge.y, &bit_challenge.z);
        let x = &poly_challenge.x;

        // Precompute some variables
        let zz = (*z) * z;
        let minus_z = -(*z);
        let z_j = util::scalar_exp_vartime(z, j as u64); // z^j
        let y_jn = util::scalar_exp_vartime(y, (j * n) as u64); // y^(j*n)
        let y_jn_inv = y_jn.inverse().unwrap(); // y^(-j*n)
        let y_inv = y.inverse().unwrap(); // y^(-1)

        if self.t_x != inner_product(&self.l_vec, &self.r_vec) {
            return Err(());
        }

        let g = self.l_vec.iter().map(|l_i| minus_z - l_i);
        let h = self
            .r_vec
            .iter()
            .zip(util::exp_iter(C::ScalarField::from(2u64)))
            .zip(util::exp_iter(y_inv))
            .map(|((r_i, exp_2), exp_y_inv)| {
                (*z) + exp_y_inv * y_jn_inv * (-(*r_i)) + exp_y_inv * y_jn_inv * (zz * z_j * exp_2)
            });

        let P_check = C::Group::msm(
            iter::once(&bit_commitment.A_j)
                .chain(iter::once(&bit_commitment.S_j))
                .chain(iter::once(&pc_gens.B_blinding))
                .chain(bp_gens.share(j).G(n))
                .chain(bp_gens.share(j).H(n))
                .copied()
                .collect::<Vec<C>>()
                .as_slice(),
            iter::once(C::ScalarField::one())
                .chain(iter::once(*x))
                .chain(iter::once(-self.e_blinding))
                .chain(g)
                .chain(h)
                .collect::<Vec<C::ScalarField>>()
                .as_slice(),
        )
        .expect("input slice lengths should match");

        if !P_check.into_affine().is_zero() {
            return Err(());
        }

        let sum_of_powers_y: C::ScalarField = util::sum_of_powers(&y, n);
        let sum_of_powers_2 = util::sum_of_powers(&C::ScalarField::from(2u64), n);
        let delta = (*z - zz) * sum_of_powers_y * y_jn - (*z) * zz * sum_of_powers_2 * z_j;

        let t_check = C::Group::msm(
            iter::once(&bit_commitment.V_j)
                .chain(iter::once(&poly_commitment.T_1_j))
                .chain(iter::once(&poly_commitment.T_2_j))
                .chain(iter::once(&pc_gens.B))
                .chain(iter::once(&pc_gens.B_blinding))
                .copied()
                .collect::<Vec<C>>()
                .as_slice(),
            iter::once(zz * z_j)
                .chain(iter::once(*x))
                .chain(iter::once(x.square()))
                .chain(iter::once(delta - self.t_x))
                .chain(iter::once(-self.t_x_blinding))
                .collect::<Vec<C::ScalarField>>()
                .as_slice(),
        )
        .expect("input slice lengths should match");

        if t_check.into_affine().is_zero() {
            Ok(())
        } else {
            Err(())
        }
    }
}
