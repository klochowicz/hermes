use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use bdk::bitcoin::util::bip32::ExtendedPrivKey;
use bdk::bitcoin::util::psbt::PartiallySignedTransaction;
use bdk::bitcoin::Address;
use bdk::bitcoin::Amount;
use bdk::bitcoin::OutPoint;
use bdk::bitcoin::PublicKey;
use bdk::bitcoin::Txid;
use bdk::blockchain::Blockchain;
use bdk::blockchain::ElectrumBlockchain;
use bdk::database::BatchDatabase;
use bdk::wallet::tx_builder::TxOrdering;
use bdk::wallet::AddressIndex;
use bdk::FeeRate;
use bdk::KeychainKind;
use bdk::SignOptions;
use bdk::SyncOptions;
use maia::PartyParams;
use maia::TxBuilderExt;
use model::Timestamp;
use model::TxFeeRate;
use model::WalletInfo;
use statrs::statistics::*;
use std::collections::HashSet;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::watch;
use tokio_tasks::Tasks;
use xtra_productivity::xtra_productivity;
use xtras::SendInterval;

const SYNC_INTERVAL: Duration = Duration::from_secs(60);

static BALANCE_GAUGE: conquer_once::Lazy<prometheus::Gauge> = conquer_once::Lazy::new(|| {
    prometheus::register_gauge!(
        "wallet_balance_satoshis",
        "The sum of available UTXOs in the wallet in satoshis."
    )
    .unwrap()
});
static NUM_UTXO_GAUGE: conquer_once::Lazy<prometheus::Gauge> = conquer_once::Lazy::new(|| {
    prometheus::register_gauge!(
        "wallet_utxos_total",
        "The number of available UTXOs in the wallet."
    )
    .unwrap()
});
static MEDIAN_UTXO_VALUE_GAUGE: conquer_once::Lazy<prometheus::Gauge> =
    conquer_once::Lazy::new(|| {
        prometheus::register_gauge!(
            "wallet_median_utxo_satoshis",
            "The median UTXO, in satoshis."
        )
        .unwrap()
    });
static MIN_UTXO_VALUE_GAUGE: conquer_once::Lazy<prometheus::Gauge> =
    conquer_once::Lazy::new(|| {
        prometheus::register_gauge!(
            "wallet_min_utxo_satoshis",
            "The smallest UTXO, in satoshis."
        )
        .unwrap()
    });
static MAX_UTXO_VALUE_GAUGE: conquer_once::Lazy<prometheus::Gauge> =
    conquer_once::Lazy::new(|| {
        prometheus::register_gauge!("wallet_max_utxo_satoshis", "The largest UTXO, in satoshis.")
            .unwrap()
    });
static MEAN_UTXO_VALUE_GAUGE: conquer_once::Lazy<prometheus::Gauge> =
    conquer_once::Lazy::new(|| {
        prometheus::register_gauge!("wallet_mean_utxo_satoshis", "The mean UTXO, in satoshis.")
            .unwrap()
    });
static STD_DEV_UTXO_VALUE_GAUGE: conquer_once::Lazy<prometheus::Gauge> =
    conquer_once::Lazy::new(|| {
        prometheus::register_gauge!(
            "wallet_stddev_utxo_satoshis",
            "The standard deviation across all UTXOs, in satoshis."
        )
        .unwrap()
    });

pub struct Actor<B> {
    wallet: bdk::Wallet<bdk::database::MemoryDatabase>,
    blockchain_client: B,
    used_utxos: LockedUtxos,
    tasks: Tasks,
    sender: watch::Sender<Option<WalletInfo>>,
}

#[derive(thiserror::Error, Debug, Clone, Copy)]
#[error("The transaction is already in the blockchain")]
pub struct TransactionAlreadyInBlockchain;

impl Actor<ElectrumBlockchain> {
    pub fn new(
        electrum_rpc_url: &str,
        ext_priv_key: ExtendedPrivKey,
    ) -> Result<(Self, watch::Receiver<Option<WalletInfo>>)> {
        let client = bdk::electrum_client::Client::new(electrum_rpc_url)
            .context("Failed to initialize Electrum RPC client")?;

        let db = bdk::database::MemoryDatabase::new();

        let wallet = bdk::Wallet::new(
            bdk::template::Bip84(ext_priv_key, KeychainKind::External),
            Some(bdk::template::Bip84(ext_priv_key, KeychainKind::Internal)),
            ext_priv_key.network,
            db,
        )?;

        // UTXOs chosen after coin selection will only be locked for a
        // few wallet sync intervals. UTXOs which were actually
        // included in published transactions should be marked as
        // spent by the internal bdk wallet by then. UTXOs which ended
        // up not being used are expected to be safe to be reused by
        // then without incurring in double spend attempts.
        let time_to_lock = SYNC_INTERVAL * 4;

        let (sender, receiver) = watch::channel(None);
        let actor = Self {
            wallet,
            tasks: Tasks::default(),
            sender,
            used_utxos: LockedUtxos::new(time_to_lock),
            blockchain_client: ElectrumBlockchain::from(client),
        };

        Ok((actor, receiver))
    }
}

impl Actor<ElectrumBlockchain> {
    fn sync_internal(&mut self) -> Result<WalletInfo> {
        let now = Instant::now();
        self.wallet
            .sync(&self.blockchain_client, SyncOptions::default())
            .context("Failed to sync wallet")?;

        let balance = self.wallet.get_balance()?;

        let utxo_values = Data::new(
            self.wallet
                .list_unspent()?
                .into_iter()
                .map(|utxo| utxo.txout.value as f64)
                .collect::<Vec<_>>(),
        );

        BALANCE_GAUGE.set(balance as f64);
        NUM_UTXO_GAUGE.set(utxo_values.len() as f64);
        MEDIAN_UTXO_VALUE_GAUGE.set(utxo_values.median());
        MIN_UTXO_VALUE_GAUGE.set(utxo_values.min());
        MAX_UTXO_VALUE_GAUGE.set(utxo_values.max());
        MEAN_UTXO_VALUE_GAUGE.set(utxo_values.mean().unwrap_or_default());
        STD_DEV_UTXO_VALUE_GAUGE.set(utxo_values.std_dev().unwrap_or_default());

        let address = self.wallet.get_address(AddressIndex::LastUnused)?.address;

        let wallet_info = WalletInfo {
            balance: Amount::from_sat(balance),
            address,
            last_updated_at: Timestamp::now(),
        };

        tracing::trace!(target : "wallet", sync_time_sec = %now.elapsed().as_secs(), "Wallet sync done");
        Ok(wallet_info)
    }
}

#[xtra_productivity]
impl Actor<ElectrumBlockchain> {
    pub fn handle_sync(&mut self, _msg: Sync) {
        let wallet_info_update = match self.sync_internal() {
            Ok(wallet_info) => Some(wallet_info),
            Err(e) => {
                tracing::debug!("{:#}", e);
                None
            }
        };
        let _ = self.sender.send(wallet_info_update);
    }

    pub fn handle_withdraw(&mut self, msg: Withdraw) -> Result<Txid> {
        self.sync_internal()?;

        if msg.address.network != self.wallet.network() {
            bail!(
                "Address has invalid network. It was {} but the wallet is connected to {}",
                msg.address.network,
                self.wallet.network()
            )
        }

        let fee_rate = msg.fee.unwrap_or_else(FeeRate::default_min_relay_fee);
        let address = msg.address;

        let mut psbt = {
            let mut tx_builder = self.wallet.build_tx();

            tx_builder
                .fee_rate(fee_rate)
                // Turn on RBF signaling
                .enable_rbf();

            match msg.amount {
                Some(amount) => {
                    tracing::info!(%amount, %address, "Withdrawing from wallet");

                    tx_builder.add_recipient(address.script_pubkey(), amount.as_sat());
                }
                None => {
                    tracing::info!(%address, "Draining wallet");

                    tx_builder.drain_wallet().drain_to(address.script_pubkey());
                }
            }

            let (psbt, _) = tx_builder.finish()?;

            psbt
        };

        self.wallet.sign(&mut psbt, SignOptions::default())?;

        let tx = psbt.extract_tx();
        let txid = tx.txid();
        self.blockchain_client.broadcast(&tx)?;

        tracing::info!(%txid, "Withdraw successful");

        Ok(txid)
    }
}

#[xtra_productivity]
impl<B> Actor<B>
where
    Self: xtra::Actor,
{
    pub fn handle_sign(&mut self, msg: Sign) -> Result<PartiallySignedTransaction> {
        let mut psbt = msg.psbt;

        self.wallet
            .sign(
                &mut psbt,
                SignOptions {
                    trust_witness_utxo: true,
                    ..Default::default()
                },
            )
            .context("could not sign transaction")?;

        Ok(psbt)
    }

    pub fn build_party_params(
        &mut self,
        BuildPartyParams {
            amount,
            identity_pk,
            fee_rate,
        }: BuildPartyParams,
    ) -> Result<PartyParams> {
        let psbt = self
            .wallet
            .build_lock_tx(amount, &mut self.used_utxos, fee_rate.into())?;

        Ok(PartyParams {
            lock_psbt: psbt,
            identity_pk,
            lock_amount: amount,
            address: self.wallet.get_address(AddressIndex::New)?.address,
        })
    }
}

#[async_trait]
impl xtra::Actor for Actor<ElectrumBlockchain> {
    type Stop = ();
    async fn started(&mut self, ctx: &mut xtra::Context<Self>) {
        let this = ctx.address().expect("self to be alive");

        // We only cache the addresses at startup
        if let Err(e) = self
            .wallet
            .ensure_addresses_cached(1000)
            .with_context(|| "Could not cache addresses")
        {
            tracing::warn!("{:#}", e);
        }

        self.tasks.add(this.send_interval(SYNC_INTERVAL, || Sync));
    }

    async fn stopped(self) -> Self::Stop {}
}

#[derive(Clone, Copy)]
pub struct BuildPartyParams {
    pub amount: Amount,
    pub identity_pk: PublicKey,
    pub fee_rate: TxFeeRate,
}

/// Private message to trigger a sync.
struct Sync;

pub struct Sign {
    pub psbt: PartiallySignedTransaction,
}

pub struct Withdraw {
    pub amount: Option<Amount>,
    pub fee: Option<FeeRate>,
    pub address: Address,
}

/// Bitcoin error codes: <https://github.com/bitcoin/bitcoin/blob/97d3500601c1d28642347d014a6de1e38f53ae4e/src/rpc/protocol.h#L23>
#[derive(Clone, Copy)]
pub enum RpcErrorCode {
    /// General error during transaction or block submission Error code -25.
    RpcVerifyError,
    /// Transaction already in chain. Error code -27.
    RpcVerifyAlreadyInChain,
}

impl From<RpcErrorCode> for i64 {
    fn from(code: RpcErrorCode) -> Self {
        match code {
            RpcErrorCode::RpcVerifyError => -25,
            RpcErrorCode::RpcVerifyAlreadyInChain => -27,
        }
    }
}

/// Module private trait to faciliate testing.
///
/// Implementing this generically on `bdk::Wallet` allows us to call it on a dummy wallet in the
/// test.
trait BuildLockTx {
    fn build_lock_tx(
        &mut self,
        amount: Amount,
        used_utxos: &mut LockedUtxos,
        fee_rate: FeeRate,
    ) -> Result<PartiallySignedTransaction>;
}

impl<D> BuildLockTx for bdk::Wallet<D>
where
    D: BatchDatabase,
{
    fn build_lock_tx(
        &mut self,
        amount: Amount,
        used_utxos: &mut LockedUtxos,
        fee_rate: FeeRate,
    ) -> Result<PartiallySignedTransaction> {
        let mut builder = self.build_tx();

        builder
            .ordering(TxOrdering::Bip69Lexicographic) // TODO: I think this is pointless but we did this in maia.
            .fee_rate(fee_rate)
            .unspendable(used_utxos.list())
            .add_2of2_multisig_recipient(amount);

        let (psbt, _) = builder.finish()?;

        let used_inputs = psbt
            .global
            .unsigned_tx
            .input
            .iter()
            .map(|input| input.previous_output);
        used_utxos.extend(used_inputs);

        Ok(psbt)
    }
}

struct LockedUtxos {
    inner: HashSet<(Instant, OutPoint)>,
    time_to_lock: Duration,
}

impl LockedUtxos {
    fn new(time_to_lock: Duration) -> Self {
        Self {
            inner: HashSet::default(),
            time_to_lock,
        }
    }

    /// Add new elements to the set of locked UTXOs.
    fn extend<T: IntoIterator<Item = OutPoint>>(&mut self, utxos: T) {
        let now = Instant::now();
        let utxos = utxos.into_iter().map(|utxo| (now, utxo));

        self.inner.extend(utxos);
    }

    /// Return the list of locked UTXOs.
    ///
    /// Before creating the list, it removes all elements which should
    /// no longer be part of the set of locked UTXOs.
    fn list(&mut self) -> Vec<OutPoint> {
        self.remove_expired();
        self.inner.iter().map(|(_, utxo)| utxo).copied().collect()
    }

    /// Remove all elements in the set of locked UTXOs which have been
    /// stored for longer than `time_to_lock`.
    fn remove_expired(&mut self) {
        let now = Instant::now();

        self.inner = self
            .inner
            .drain()
            .skip_while(|(locked_at, _)| now >= *locked_at + self.time_to_lock)
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bdk_ext::keypair;
    use bdk_ext::new_test_wallet;
    use itertools::Itertools;
    use rand::thread_rng;
    use std::collections::HashSet;
    use xtra::Actor as _;

    impl Actor<()> {
        pub fn new_offline(
            utxo_amount: Amount,
            num_utxos: u8,
            time_to_lock: Duration,
        ) -> Result<Self> {
            let wallet = new_test_wallet(&mut thread_rng(), utxo_amount, num_utxos)?;

            let (sender, _receiver) = watch::channel(None);

            Ok(Self {
                wallet,
                tasks: Tasks::default(),
                sender,
                used_utxos: LockedUtxos {
                    inner: HashSet::default(),
                    time_to_lock,
                },
                blockchain_client: (),
            })
        }
    }

    #[async_trait]
    impl xtra::Actor for Actor<()> {
        type Stop = ();

        async fn stopped(self) -> Self::Stop {}
    }

    #[test]
    fn creating_two_lock_transactions_uses_different_utxos() {
        let mut wallet = new_test_wallet(&mut thread_rng(), Amount::from_sat(1000), 10).unwrap();
        let mut used_utxos = LockedUtxos {
            inner: HashSet::default(),
            time_to_lock: Duration::from_secs(120),
        };

        let lock_tx_1 = wallet
            .build_lock_tx(
                Amount::from_sat(2500),
                &mut used_utxos,
                FeeRate::default_min_relay_fee(),
            )
            .unwrap();
        let lock_tx_2 = wallet
            .build_lock_tx(
                Amount::from_sat(2500),
                &mut used_utxos,
                FeeRate::default_min_relay_fee(),
            )
            .unwrap();

        let mut utxos_in_transaction = HashSet::new();
        utxos_in_transaction.extend(
            lock_tx_1
                .global
                .unsigned_tx
                .input
                .iter()
                .map(|i| i.previous_output),
        );
        utxos_in_transaction.extend(
            lock_tx_2
                .global
                .unsigned_tx
                .input
                .iter()
                .map(|i| i.previous_output),
        );

        // 2 TX a 2500 sats with UTXOs worth 1000s = 6 inputs
        // If there are 6 UTXOs in the HashSet, we know that they are all different (HashSets don't
        // allow duplicates!)
        let expected_num_utxos = 6;

        assert_eq!(utxos_in_transaction.len(), expected_num_utxos);
        assert_eq!(
            utxos_in_transaction.iter().sorted().collect::<Vec<_>>(),
            used_utxos.list().iter().sorted().collect::<Vec<_>>(),
        );
    }

    #[tokio::test]
    async fn utxo_is_locked_after_building_party_params() {
        let mut tasks = Tasks::default();

        // create wallet with only one UTXO which will be locked for a
        // long time after being used
        let actor = Actor::new_offline(Amount::ONE_BTC, 1, Duration::from_secs(120))
            .unwrap()
            .create(None)
            .spawn(&mut tasks);

        let (_, identity_pk) = keypair::new(&mut thread_rng());

        // building party params locks our only UTXO
        actor
            .send(BuildPartyParams {
                amount: Amount::from_btc(0.2).unwrap(),
                identity_pk,
                fee_rate: TxFeeRate::default(),
            })
            .await
            .unwrap()
            .expect("single UTXO to be available");

        // our only UTXO remains locked, so the second attempt at
        // building party params fails
        actor
            .send(BuildPartyParams {
                amount: Amount::from_btc(0.2).unwrap(),
                identity_pk,
                fee_rate: TxFeeRate::default(),
            })
            .await
            .unwrap()
            .expect_err("single UTXO to remain locked");
    }

    #[tokio::test]
    async fn utxo_can_be_unlocked_after_marking_as_unspendable() {
        let mut tasks = Tasks::default();

        // create wallet with only one UTXO which will be locked for a
        // few seconds after being used
        let time_to_lock = Duration::from_secs(2);
        let actor = Actor::new_offline(Amount::ONE_BTC, 1, time_to_lock)
            .unwrap()
            .create(None)
            .spawn(&mut tasks);

        let (_, identity_pk) = keypair::new(&mut thread_rng());

        // building party params locks our only UTXO
        actor
            .send(BuildPartyParams {
                amount: Amount::from_btc(0.2).unwrap(),
                identity_pk,
                fee_rate: TxFeeRate::default(),
            })
            .await
            .unwrap()
            .expect("single UTXO to be available");

        // our only UTXO remains locked, so the second attempt at
        // building party params fails
        actor
            .send(BuildPartyParams {
                amount: Amount::from_btc(0.2).unwrap(),
                identity_pk,
                fee_rate: TxFeeRate::default(),
            })
            .await
            .unwrap()
            .expect_err("single UTXO to remain locked");

        // wait for lock on UTXO to expire
        tokio::time::sleep(time_to_lock).await;

        // after enough time has passed, our UTXO can once again be
        // used to build party params
        let _party_params = actor
            .send(BuildPartyParams {
                amount: Amount::from_btc(0.2).unwrap(),
                identity_pk,
                fee_rate: TxFeeRate::default(),
            })
            .await
            .unwrap()
            .expect("single UTXO to be available after unlocking it");
    }
}
