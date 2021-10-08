use crate::model::{Leverage, Usd};
use crate::payout_curve::curve::Curve;

use anyhow::{Context, Result};
use bdk::bitcoin;
use cfd_protocol::{generate_payouts, Payout};
use ndarray::prelude::*;
use num::{FromPrimitive, ToPrimitive};
use rust_decimal::Decimal;

mod basis;
mod basis_eval;
mod compat;
mod csr_tools;
mod curve;
mod curve_factory;
mod splineobject;
mod utils;

pub enum LongPosition {
    #[allow(dead_code)]
    Maker,
    Taker,
}

/// function to generate an iterator of values, heuristically viewed as:
///
///     `[left_price_boundary, right_price_boundary], maker_payout_value`
///
/// with units
///
///     `[Usd, Usd], bitcoin::Amount`
///
/// A key item to note is that although the POC logic has been to imposed
/// that maker goes short every time, there is no reason to make the math
/// have this imposition as well. As such, the `long_position` parameter
/// is used to indicate which party (Maker or Taker) has the long position,
/// and everything else is handled internally.
///
/// As well, the POC has also demanded that the Maker always has unity
/// leverage, hence why the ability to to specify this amount has been
/// omitted from the parameters. Internally, it is hard-coded to unity
/// in the call to PayoutCurve::new(), so this behaviour can be changed in
/// the future trivially.
///
/// ### paramters
/// * price: BTC-USD exchange rate used to create CFD contract
/// * quantity: Interger number of one-dollar USD contracts contained in the
/// CFD; expressed as a Usd amount
/// * long_position: Indicates which party (Maker or Taker) has taken the long
/// position for the CFD
/// * taker_leverage: Leveraging used by the taker
///
/// ### returns
/// * tuple: first element is the iterator described above, second element is
/// the total amount of BTC that can change ownership via this model. That is,
/// `taker_payout = total - maker_payout`
pub fn calculate(
    price: Usd,
    quantity: Usd,
    long_position: LongPosition,
    taker_leverage: Leverage,
) -> Result<(Vec<Payout>, bitcoin::Amount)> {
    let contract_value = 1_f64;
    let initial_rate = price
        .try_into_u64()
        .context("Cannot convert price to u64")? as f64;
    let quantity = quantity
        .try_into_u64()
        .context("Cannot convert quantity to u64")? as usize;
    let maker_leverage = 1;

    let maker_payout_curve = match long_position {
        LongPosition::Maker => PayoutCurve::new(
            initial_rate as f64,
            maker_leverage,
            taker_leverage.0 as usize,
            quantity,
            contract_value,
            true,
            None,
        )?,
        LongPosition::Taker => PayoutCurve::new(
            initial_rate as f64,
            taker_leverage.0 as usize,
            maker_leverage,
            quantity,
            contract_value,
            false,
            None,
        )?,
    };

    let n_payouts = 200;
    let payouts_arr = maker_payout_curve.generate_payout_scheme(n_payouts)?;
    let mut payouts_vec = Vec::<_>::with_capacity(n_payouts);

    for (i, e) in payouts_arr.slice(s![..n_payouts, 0]).iter().enumerate() {
        let bnd_left = *e as u64;
        let bnd_right = payouts_arr[[i, 1]] as u64;

        let maker_amount = payouts_arr[[i, 2]];

        let taker_amount = to_bitcoin_amount(maker_payout_curve.total_value - maker_amount)?;
        let maker_amount = to_bitcoin_amount(maker_amount)?;

        // This printout is useful for updating the snapshot test.
        // println!(
        //     "payouts({}..={}, {}, {}),",
        //     bnd_left,
        //     bnd_right,
        //     maker_amount.as_sat(),
        //     taker_amount.as_sat()
        // );

        let elem = generate_payouts(bnd_left..=bnd_right, maker_amount, taker_amount)?;
        payouts_vec.extend(elem);
    }
    let btc_total = to_bitcoin_amount(maker_payout_curve.total_value)?;

    Ok((payouts_vec, btc_total))
}

/// Converts a float with any precision to a [`bitcoin::Amount`].
fn to_bitcoin_amount(btc: f64) -> Result<bitcoin::Amount> {
    let sats_per_btc = Decimal::from(100_000_000);

    let btc = Decimal::from_f64(btc).context("Cannot create decimal from float")?;
    let sats = btc * sats_per_btc;
    let sats = sats.to_u64().context("Cannot fit sats into u64")?;

    Ok(bitcoin::Amount::from_sat(sats))
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to init CSR object--is the specified shape correct?")]
    #[allow(clippy::upper_case_acronyms)]
    CannotInitCSR,
    #[error("matrix must be square")]
    MatrixMustBeSquare,
    #[error("evaluation outside parametric domain")]
    InvalidDomain,
    #[error("einsum error--array size mismatch?")]
    Einsum,
    #[error("no operand string found")]
    NoEinsumOperatorString,
    #[error("cannot connect periodic curves")]
    CannotConnectPeriodicCurves,
    #[error("degree must be strictly positive")]
    DegreeMustBePositive,
    #[error("all parameter arrays must have the same length if not using a tensor grid")]
    InvalidDerivative,
    #[error("Rational derivative not implemented for order sum(d) > 1")]
    DerivativeNotImplemented,
    #[error("requested segmentation is too coarse for this curve")]
    InvalidSegmentation,
    #[error("concatonation error")]
    NdArray {
        #[from]
        source: ndarray::ShapeError,
    },
    #[error(transparent)]
    NotOneDimensional {
        #[from]
        source: compat::NotOneDimensional,
    },
}

#[derive(Clone, Debug)]
struct PayoutCurve {
    pub curve: Curve,
    pub has_upper_limit: bool,
    pub lower_corner: f64,
    pub upper_corner: f64,
    pub total_value: f64,
}

impl PayoutCurve {
    pub fn new(
        initial_rate: f64,
        leverage_long: usize,
        leverage_short: usize,
        n_contracts: usize,
        contract_value: f64,
        is_long: bool,
        tolerance: Option<f64>,
    ) -> Result<Self, Error> {
        let tolerance = tolerance.unwrap_or(1e-6);
        let bounds = cutoffs(initial_rate, leverage_long, leverage_short);
        let total_value = pool_value(
            initial_rate,
            n_contracts,
            contract_value,
            leverage_long,
            leverage_short,
        );
        let mut curve = curve_factory::line((0., 0.), (bounds.0, 0.), false)?;

        if is_long {
            let payout = create_long_payout_function(
                initial_rate,
                n_contracts,
                contract_value,
                leverage_long,
            );
            let variable_payout =
                curve_factory::fit(payout, bounds.0, bounds.1, Some(tolerance), None)?;
            curve.append(variable_payout)?;
        } else {
            let payout = create_short_payout_function(
                initial_rate,
                n_contracts,
                contract_value,
                leverage_short,
            );
            let variable_payout =
                curve_factory::fit(payout, bounds.0, bounds.1, Some(tolerance), None)?;
            curve.append(variable_payout)?;
        }

        let upper_corner;
        if bounds.2 {
            let upper_liquidation = curve_factory::line(
                (bounds.1, total_value),
                (4. * initial_rate, total_value),
                false,
            )?;
            curve.append(upper_liquidation)?;
            upper_corner = bounds.1;
        } else {
            upper_corner = curve.spline.bases[0].end();
        }

        Ok(PayoutCurve {
            curve,
            has_upper_limit: bounds.2,
            lower_corner: bounds.0,
            upper_corner,
            total_value,
        })
    }

    pub fn generate_payout_scheme(&self, n_segments: usize) -> Result<Array2<f64>, Error> {
        let n_min;
        if self.has_upper_limit {
            n_min = 3;
        } else {
            n_min = 2;
        }

        if n_segments < n_min {
            return Result::Err(Error::InvalidSegmentation);
        }

        let t;
        if self.has_upper_limit {
            t = self.build_sampling_vector_upper_bounded(n_segments);
        } else {
            t = self.build_sampling_vector_upper_unbounded(n_segments)
        }

        let mut z_arr = self.curve.evaluate(&mut &[t][..])?;
        if self.has_upper_limit {
            self.modify_samples_bounded(&mut z_arr);
        } else {
            self.modify_samples_unbounded(&mut z_arr);
        }
        self.generate_segments(&mut z_arr);

        Ok(z_arr)
    }

    fn build_sampling_vector_upper_bounded(&self, n_segs: usize) -> Array1<f64> {
        let knots = &self.curve.spline.knots(0, None).unwrap()[0];
        let klen = knots.len();
        let n_64 = (n_segs + 1) as f64;
        let d = knots[klen - 2] - knots[1];
        let delta_0 = d / (2. * (n_64 - 5.));
        let delta_1 = d * (n_64 - 6.) / ((n_64 - 5.) * (n_64 - 4.));

        let mut vec = Vec::<f64>::with_capacity(n_segs + 2);
        for i in 0..n_segs + 2 {
            if i == 0 {
                vec.push(self.curve.spline.bases[0].start());
            } else if i == 1 {
                vec.push(knots[1]);
            } else if i == 2 {
                vec.push(knots[1] + delta_0);
            } else if i == n_segs - 1 {
                vec.push(knots[klen - 2] - delta_0);
            } else if i == n_segs {
                vec.push(knots[klen - 2]);
            } else if i == n_segs + 1 {
                vec.push(self.curve.spline.bases[0].end());
            } else {
                let c = (i - 2) as f64;
                vec.push(knots[1] + delta_0 + c * delta_1);
            }
        }
        Array1::<f64>::from_vec(vec)
    }

    fn build_sampling_vector_upper_unbounded(&self, n_segs: usize) -> Array1<f64> {
        let knots = &self.curve.spline.knots(0, None).unwrap()[0];
        let klen = knots.len();
        let n_64 = (n_segs + 1) as f64;
        let d = knots[klen - 1] - knots[1];
        let delta = d / (n_64 - 1_f64);
        let delta_x = d / (2. * (n_64 - 1_f64));
        let delta_y = 3. * d / (2. * (n_64 - 1_f64));

        let mut vec = Vec::<f64>::with_capacity(n_segs + 2);
        for i in 0..n_segs + 2 {
            if i == 0 {
                vec.push(self.curve.spline.bases[0].start());
            } else if i == 1 {
                vec.push(knots[1]);
            } else if i == 2 {
                vec.push(knots[1] + delta_x);
            } else if i == n_segs {
                vec.push(knots[klen - 1] - delta_y);
            } else if i == n_segs + 1 {
                vec.push(knots[klen - 1]);
            } else {
                let c = (i - 2) as f64;
                vec.push(knots[1] + delta_x + c * delta);
            }
        }
        Array1::<f64>::from_vec(vec)
    }

    fn modify_samples_bounded(&self, arr: &mut Array2<f64>) {
        let n = arr.shape()[0];
        let capacity = 2 * (n - 2);
        let mut vec = Vec::<f64>::with_capacity(2 * capacity);
        for (i, e) in arr.slice(s![.., 0]).iter().enumerate() {
            if i < 2 || i > n - 3 {
                vec.push(*e);
            } else if i == 2 {
                vec.push(arr[[i - 1, 0]]);
                vec.push(arr[[i, 1]]);
                vec.push((*e + arr[[i + 1, 0]]) / 2.);
            } else if i == n - 3 {
                vec.push((arr[[i - 1, 0]] + *e) / 2.);
                vec.push(arr[[i, 1]]);
                vec.push(arr[[i + 1, 0]]);
            } else {
                vec.push((arr[[i - 1, 0]] + *e) / 2.);
                vec.push(arr[[i, 1]]);
                vec.push((*e + arr[[i + 1, 0]]) / 2.);
            }
            vec.push(arr[[i, 1]]);
        }

        *arr = Array2::<f64>::from_shape_vec((capacity, 2), vec).unwrap();
    }

    fn modify_samples_unbounded(&self, arr: &mut Array2<f64>) {
        let n = arr.shape()[0];
        let capacity = 2 * (n - 1);
        let mut vec = Vec::<f64>::with_capacity(2 * capacity);
        for (i, e) in arr.slice(s![.., 0]).iter().enumerate() {
            if i < 2 {
                vec.push(*e);
            } else if i == 2 {
                vec.push(arr[[i - 1, 0]]);
                vec.push(arr[[i, 1]]);
                vec.push((*e + arr[[i + 1, 0]]) / 2.);
            } else if i == n - 1 {
                vec.push((arr[[i - 1, 0]] + *e) / 2.);
                vec.push(arr[[i, 1]]);
                vec.push(arr[[i, 0]]);
            } else {
                vec.push((arr[[i - 1, 0]] + *e) / 2.);
                vec.push(arr[[i, 1]]);
                vec.push((*e + arr[[i + 1, 0]]) / 2.);
            }
            vec.push(arr[[i, 1]]);
        }

        *arr = Array2::<f64>::from_shape_vec((capacity, 2), vec).unwrap();
    }

    /// this should only be used on an array `arr` that has been
    /// processed by self.modify_samples_* first, otherwise the results
    /// will be jibberish.
    fn generate_segments(&self, arr: &mut Array2<f64>) {
        let capacity = 3 * arr.shape()[0] / 2;
        let mut vec = Vec::<f64>::with_capacity(capacity);
        for (i, e) in arr.slice(s![.., 0]).iter().enumerate() {
            if i == 0 {
                vec.push(e.floor());
            } else if i % 2 == 1 {
                vec.push(e.floor());
                vec.push(arr[[i, 1]]);
            } else if (e.ceil() - vec[vec.len() - 2]).abs() < 1e-3 {
                vec.push(e.ceil() + 1_f64);
            } else {
                vec.push(e.ceil());
            }
        }

        *arr = Array2::<f64>::from_shape_vec((capacity / 3, 3), vec).unwrap();
    }
}

fn cutoffs(initial_rate: f64, leverage_long: usize, leverage_short: usize) -> (f64, f64, bool) {
    let ll_64 = leverage_long as f64;
    let ls_64 = leverage_short as f64;
    let a = initial_rate * ll_64 / (ll_64 + 1_f64);
    if leverage_short == 1 {
        let b = 2. * initial_rate;
        return (a, b, false);
    }
    let b = initial_rate * ls_64 / (ls_64 - 1_f64);

    (a, b, true)
}

fn pool_value(
    initial_rate: f64,
    n_contracts: usize,
    contract_value: f64,
    leverage_long: usize,
    leverage_short: usize,
) -> f64 {
    let ll_64 = leverage_long as f64;
    let ls_64 = leverage_short as f64;
    let n_64 = n_contracts as f64;

    (n_64 * contract_value / initial_rate) * (1_f64 / ll_64 + 1_f64 / ls_64)
}

fn create_long_payout_function(
    initial_rate: f64,
    n_contracts: usize,
    contract_value: f64,
    leverage_long: usize,
) -> impl Fn(&Array1<f64>) -> Array2<f64> {
    let n_64 = n_contracts as f64;
    let ll_64 = leverage_long as f64;

    move |t: &Array1<f64>| {
        let mut vec = Vec::<f64>::with_capacity(2 * t.len());
        for e in t.iter() {
            let eval = (n_64 * contract_value)
                * (1_f64 / (initial_rate * ll_64) + (1_f64 / initial_rate - 1_f64 / e));
            vec.push(*e);
            vec.push(eval);
        }

        Array2::<f64>::from_shape_vec((t.len(), 2), vec).unwrap()
    }
}

fn create_short_payout_function(
    initial_rate: f64,
    n_contracts: usize,
    contract_value: f64,
    leverage_short: usize,
) -> impl Fn(&Array1<f64>) -> Array2<f64> {
    let n_64 = n_contracts as f64;
    let ls_64 = leverage_short as f64;

    move |t: &Array1<f64>| {
        let mut vec = Vec::<f64>::with_capacity(2 * t.len());
        for e in t.iter() {
            let eval = (n_64 * contract_value)
                * (1_f64 / (initial_rate * ls_64) - (1_f64 / initial_rate - 1_f64 / e));
            vec.push(*e);
            vec.push(eval);
        }

        Array2::<f64>::from_shape_vec((t.len(), 2), vec).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use std::ops::RangeInclusive;

    #[test]
    fn test_bounded() {
        let initial_rate = 40000.0;
        let leverage_long = 5;
        let leverage_short = 2;
        let n_contracts = 200;
        let contract_value = 100.;
        let is_long = true;

        let payout = PayoutCurve::new(
            initial_rate,
            leverage_long,
            leverage_short,
            n_contracts,
            contract_value,
            is_long,
            None,
        )
        .unwrap();

        let z = payout.generate_payout_scheme(5000).unwrap();

        assert!(z.shape()[0] == 5000);
    }

    #[test]
    fn test_unbounded() {
        let initial_rate = 40000.0;
        let leverage_long = 5;
        let leverage_short = 1;
        let n_contracts = 200;
        let contract_value = 100.;
        let is_long = true;

        let payout = PayoutCurve::new(
            initial_rate,
            leverage_long,
            leverage_short,
            n_contracts,
            contract_value,
            is_long,
            None,
        )
        .unwrap();

        let z = payout.generate_payout_scheme(5000).unwrap();

        // out-by-one error expected at this point in time
        assert!(z.shape()[0] == 5001);
    }

    #[test]
    fn calculate_snapshot() {
        let (actual_payouts, _) = calculate(
            Usd(dec!(54000.00)),
            Usd(dec!(3500.00)),
            LongPosition::Taker,
            Leverage(5),
        )
        .unwrap();

        let expected_payouts = vec![
            payouts(0..=45000, 0, 7777777),
            payouts(45001..=45314, 261017, 7516760),
            payouts(45315..=45630, 762064, 7015713),
            payouts(45631..=45945, 1235726, 6542051),
            payouts(45946..=46260, 1682724, 6095053),
            payouts(46261..=46575, 2103777, 5674000),
            payouts(46576..=46890, 2499607, 5278170),
            payouts(46891..=47205, 2870934, 4906843),
            payouts(47206..=47520, 3218477, 4559300),
            payouts(47521..=47835, 3542958, 4234819),
            payouts(47836..=48150, 3845097, 3932680),
            payouts(48151..=48465, 4125613, 3652163),
            payouts(48466..=48780, 4385228, 3392548),
            payouts(48781..=49095, 4624662, 3153115),
            payouts(49096..=49410, 4844635, 2933142),
            payouts(49411..=49725, 5045868, 2731909),
            payouts(49726..=50040, 5229080, 2548697),
            payouts(50041..=50355, 5394992, 2382785),
            payouts(50356..=50670, 5544325, 2233452),
            payouts(50671..=50984, 5677799, 2099978),
            payouts(50985..=51299, 5796134, 1981643),
            payouts(51300..=51615, 5900050, 1877727),
            payouts(51616..=51930, 5990268, 1787508),
            payouts(51931..=52245, 6067509, 1710268),
            payouts(52246..=52560, 6132492, 1645285),
            payouts(52561..=52875, 6185938, 1591838),
            payouts(52876..=53190, 6228568, 1549209),
            payouts(53191..=53505, 6261101, 1516676),
            payouts(53506..=53820, 6284258, 1493519),
            payouts(53821..=54135, 6298759, 1479017),
            payouts(54136..=54450, 6305325, 1472451),
            payouts(54451..=54764, 6304677, 1473100),
            payouts(54765..=55080, 6297533, 1480244),
            payouts(55081..=55395, 6284615, 1493161),
            payouts(55396..=55710, 6266644, 1511133),
            payouts(55711..=56025, 6244339, 1533438),
            payouts(56026..=56340, 6218420, 1559356),
            payouts(56341..=56655, 6189609, 1588168),
            payouts(56656..=56970, 6158625, 1619152),
            payouts(56971..=57285, 6126189, 1651587),
            payouts(57286..=57600, 6093022, 1684755),
            payouts(57601..=57915, 6059827, 1717949),
            payouts(57916..=58230, 6026965, 1750812),
            payouts(58231..=58545, 5994445, 1783332),
            payouts(58546..=58860, 5962264, 1815512),
            payouts(58861..=59175, 5930419, 1847358),
            payouts(59176..=59490, 5898905, 1878872),
            payouts(59491..=59805, 5867718, 1910059),
            payouts(59806..=60120, 5836855, 1940922),
            payouts(60121..=60435, 5806311, 1971465),
            payouts(60436..=60750, 5776084, 2001693),
            payouts(60751..=61065, 5746168, 2031608),
            payouts(61066..=61380, 5716561, 2061216),
            payouts(61381..=61695, 5687258, 2090519),
            payouts(61696..=62010, 5658255, 2119522),
            payouts(62011..=62325, 5629549, 2148228),
            payouts(62326..=62640, 5601135, 2176642),
            payouts(62641..=62955, 5573010, 2204767),
            payouts(62956..=63270, 5545170, 2232607),
            payouts(63271..=63585, 5517611, 2260165),
            payouts(63586..=63900, 5490330, 2287447),
            payouts(63901..=64215, 5463321, 2314455),
            payouts(64216..=64530, 5436583, 2341194),
            payouts(64531..=64845, 5410109, 2367667),
            payouts(64846..=65160, 5383898, 2393879),
            payouts(65161..=65475, 5357944, 2419833),
            payouts(65476..=65790, 5332245, 2445532),
            payouts(65791..=66105, 5306795, 2470982),
            payouts(66106..=66420, 5281592, 2496185),
            payouts(66421..=66735, 5256631, 2521146),
            payouts(66736..=67050, 5231909, 2545868),
            payouts(67051..=67365, 5207421, 2570356),
            payouts(67366..=67680, 5183164, 2594612),
            payouts(67681..=67995, 5159135, 2618642),
            payouts(67996..=68310, 5135328, 2642449),
            payouts(68311..=68625, 5111740, 2666037),
            payouts(68626..=68940, 5088368, 2689409),
            payouts(68941..=69255, 5065207, 2712569),
            payouts(69256..=69570, 5042254, 2735523),
            payouts(69571..=69885, 5019505, 2758272),
            payouts(69886..=70200, 4996955, 2780821),
            payouts(70201..=70515, 4974602, 2803175),
            payouts(70516..=70830, 4952442, 2825335),
            payouts(70831..=71145, 4930473, 2847304),
            payouts(71146..=71460, 4908694, 2869083),
            payouts(71461..=71775, 4887102, 2890675),
            payouts(71776..=72090, 4865695, 2912081),
            payouts(72091..=72405, 4844473, 2933304),
            payouts(72406..=72720, 4823433, 2954344),
            payouts(72721..=73035, 4802573, 2975204),
            payouts(73036..=73350, 4781891, 2995886),
            payouts(73351..=73665, 4761385, 3016391),
            payouts(73666..=73980, 4741054, 3036722),
            payouts(73981..=74295, 4720896, 3056881),
            payouts(74296..=74610, 4700909, 3076868),
            payouts(74611..=74925, 4681090, 3096686),
            payouts(74926..=75240, 4661439, 3116338),
            payouts(75241..=75555, 4641953, 3135824),
            payouts(75556..=75870, 4622630, 3155146),
            payouts(75871..=76185, 4603469, 3174307),
            payouts(76186..=76500, 4584468, 3193309),
            payouts(76501..=76815, 4565624, 3212153),
            payouts(76816..=77130, 4546937, 3230840),
            payouts(77131..=77445, 4528403, 3249374),
            payouts(77446..=77760, 4510022, 3267755),
            payouts(77761..=78075, 4491791, 3285986),
            payouts(78076..=78390, 4473708, 3304068),
            payouts(78391..=78705, 4455773, 3322004),
            payouts(78706..=79020, 4437982, 3339795),
            payouts(79021..=79335, 4420333, 3357443),
            payouts(79336..=79650, 4402827, 3374950),
            payouts(79651..=79965, 4385459, 3392318),
            payouts(79966..=80280, 4368228, 3409548),
            payouts(80281..=80595, 4351133, 3426643),
            payouts(80596..=80910, 4334172, 3443605),
            payouts(80911..=81225, 4317343, 3460434),
            payouts(81226..=81540, 4300643, 3477134),
            payouts(81541..=81855, 4284071, 3493705),
            payouts(81856..=82170, 4267626, 3510151),
            payouts(82171..=82485, 4251305, 3526472),
            payouts(82486..=82800, 4235107, 3542670),
            payouts(82801..=83115, 4219029, 3558748),
            payouts(83116..=83430, 4203070, 3574707),
            payouts(83431..=83745, 4187229, 3590547),
            payouts(83746..=84060, 4171506, 3606271),
            payouts(84061..=84375, 4155899, 3621878),
            payouts(84376..=84690, 4140406, 3637371),
            payouts(84691..=85005, 4125028, 3652749),
            payouts(85006..=85320, 4109763, 3668014),
            payouts(85321..=85635, 4094610, 3683167),
            payouts(85636..=85950, 4079567, 3698209),
            payouts(85951..=86265, 4064635, 3713142),
            payouts(86266..=86580, 4049812, 3727965),
            payouts(86581..=86895, 4035096, 3742680),
            payouts(86896..=87210, 4020488, 3757289),
            payouts(87211..=87525, 4005985, 3771792),
            payouts(87526..=87840, 3991587, 3786189),
            payouts(87841..=88155, 3977293, 3800484),
            payouts(88156..=88470, 3963102, 3814675),
            payouts(88471..=88785, 3949013, 3828764),
            payouts(88786..=89100, 3935024, 3842753),
            payouts(89101..=89415, 3921135, 3856642),
            payouts(89416..=89730, 3907344, 3870432),
            payouts(89731..=90045, 3893652, 3884125),
            payouts(90046..=90360, 3880056, 3897721),
            payouts(90361..=90675, 3866555, 3911221),
            payouts(90676..=90990, 3853150, 3924627),
            payouts(90991..=91305, 3839837, 3937940),
            payouts(91306..=91620, 3826618, 3951159),
            payouts(91621..=91935, 3813489, 3964287),
            payouts(91936..=92250, 3800452, 3977325),
            payouts(92251..=92565, 3787504, 3990273),
            payouts(92566..=92880, 3774644, 4003133),
            payouts(92881..=93195, 3761872, 4015905),
            payouts(93196..=93510, 3749186, 4028591),
            payouts(93511..=93825, 3736585, 4041192),
            payouts(93826..=94140, 3724069, 4053708),
            payouts(94141..=94455, 3711636, 4066141),
            payouts(94456..=94770, 3699286, 4078491),
            payouts(94771..=95085, 3687016, 4090760),
            payouts(95086..=95400, 3674827, 4102949),
            payouts(95401..=95715, 3662718, 4115059),
            payouts(95716..=96030, 3650686, 4127091),
            payouts(96031..=96345, 3638733, 4139044),
            payouts(96346..=96660, 3626856, 4150920),
            payouts(96661..=96975, 3615057, 4162720),
            payouts(96976..=97290, 3603333, 4174444),
            payouts(97291..=97605, 3591684, 4186093),
            payouts(97606..=97920, 3580110, 4197666),
            payouts(97921..=98234, 3568611, 4209166),
            payouts(98235..=98550, 3557184, 4220593),
            payouts(98551..=98865, 3545831, 4231946),
            payouts(98866..=99179, 3534550, 4243227),
            payouts(99180..=99494, 3523340, 4254437),
            payouts(99495..=99810, 3512201, 4265576),
            payouts(99811..=100125, 3501133, 4276644),
            payouts(100126..=100439, 3490134, 4287643),
            payouts(100440..=100754, 3479205, 4298572),
            payouts(100755..=101069, 3468344, 4309433),
            payouts(101070..=101384, 3457551, 4320226),
            payouts(101385..=101699, 3446825, 4330952),
            payouts(101700..=102014, 3436165, 4341611),
            payouts(102015..=102329, 3425572, 4352205),
            payouts(102330..=102644, 3415044, 4362732),
            payouts(102645..=102959, 3404581, 4373196),
            payouts(102960..=103274, 3394182, 4383594),
            payouts(103275..=103589, 3383847, 4393930),
            payouts(103590..=103904, 3373574, 4404202),
            payouts(103905..=104219, 3363364, 4414412),
            payouts(104220..=104534, 3353216, 4424561),
            payouts(104535..=104849, 3343128, 4434648),
            payouts(104850..=105164, 3333101, 4444675),
            payouts(105165..=105479, 3323134, 4454643),
            payouts(105480..=105795, 3313226, 4464551),
            payouts(105796..=106110, 3303377, 4474400),
            payouts(106111..=106424, 3293585, 4484192),
            payouts(106425..=106739, 3283851, 4493926),
            payouts(106740..=107055, 3274174, 4503603),
            payouts(107056..=107370, 3264552, 4513225),
            payouts(107371..=107763, 3254986, 4522790),
        ]
        .concat();

        assert_eq!(actual_payouts, expected_payouts);
    }

    #[test]
    fn verfiy_tails() {
        let (actual_payouts, _) = calculate(
            Usd(dec!(54000.00)),
            Usd(dec!(3500.00)),
            LongPosition::Taker,
            Leverage(5),
        )
        .unwrap();

        // first element
        // Payout {
        //     digits: Digits(00000),
        //     maker_amount: Amount(0.00000000 BTC),
        //     taker_amount: Amount(0.07777777 BTC)
        // }
        let temp = payouts(0..=45000, 0, 7777777);
        let lower_tail = temp.first();

        // last element
        // Payout {
        //     digits: Digits(000110100100111100),
        //     maker_amount: Amount(0.03254986 BTC),
        //     taker_amount: Amount(0.04522790 BTC)
        // }
        let temp = payouts(107371..=107763, 3254986, 4522790);
        let upper_tail = temp.last();

        assert_eq!(actual_payouts.first(), lower_tail);
        assert_eq!(actual_payouts.last(), upper_tail);
    }

    fn payouts(range: RangeInclusive<u64>, maker_amount: u64, taker_amount: u64) -> Vec<Payout> {
        generate_payouts(
            range,
            bitcoin::Amount::from_sat(maker_amount),
            bitcoin::Amount::from_sat(taker_amount),
        )
        .unwrap()
    }
}
