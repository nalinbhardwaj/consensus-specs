# Whisk -- The Beacon Chain

**Notice**: This document is a work-in-progress for researchers and implementers.
## Table of contents

<!-- TOC -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Whisk -- The Beacon Chain](#whisk----the-beacon-chain)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Constants](#constants)
  - [Cryptography](#cryptography)
    - [BLS](#bls)
    - [Curdleproofs and opening proofs](#curdleproofs-and-opening-proofs)
  - [Epoch processing](#epoch-processing)
    - [`WhiskTracker`](#whisktracker)
    - [`Validator`](#validator)
    - [`BeaconState`](#beaconstate)
  - [Block processing](#block-processing)
    - [Block header](#block-header)
      - [`BeaconBlock`](#beaconblock)
    - [Whisk](#whisk)
      - [`BeaconBlockBody`](#beaconblockbody)
    - [Deposits](#deposits)
    - [`get_beacon_proposer_index`](#get_beacon_proposer_index)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- /TOC -->

## Introduction

This document details the beacon chain additions and changes of to support the Whisk SSLE,
building upon the [phase0](../phase0/beacon-chain.md) specification.

## Constants

| Name                               | Value                      | Description                                                 |
| ---------------------------------- | -------------------------- | ----------------------------------------------------------- |
| `WHISK_CANDIDATE_TRACKERS_COUNT`   | `uint64(2**14)` (= 16,384) | number of candidate trackers                                |
| `WHISK_PROPOSER_TRACKERS_COUNT`    | `uint64(2**13)` (= 8,192)  | number of proposer trackers                                 |
| `WHISK_EPOCHS_PER_SHUFFLING_PHASE` | `Epoch(2**8)` (= 256)      | epochs per shuffling phase                                  |
| `WHISK_VALIDATORS_PER_SHUFFLE`     | `uint64(2**7)` (= 128)     | number of validators shuffled per shuffle step              |
| `WHISK_PROPOSER_SELECTION_GAP`     | `Epoch(2)`                 | gap between proposer selection and the block proposal phase |
| `WHISK_MAX_SHUFFLE_PROOF_SIZE`     | `uint64(2**15)`            | max size of a shuffle proof                                 |
| `WHISK_MAX_OPENING_PROOF_SIZE`     | `uint64(2**10)`            | max size of a opening proof                                 |

| Name                               | Value                      |
| ---------------------------------- | -------------------------- |
| `DOMAIN_WHISK_CANDIDATE_SELECTION` | `DomainType('0x07000000')` |
| `DOMAIN_WHISK_SHUFFLE`             | `DomainType('0x08000000')` |
| `DOMAIN_WHISK_PROPOSER_SELECTION`  | `DomainType('0x09000000')` |

## Cryptography

### BLS

| Name              | SSZ equivalent | Description                   |
| ----------------- | -------------- | ----------------------------- |
| `BLSFieldElement` | `uint256`      | BLS12-381 scalar              |
| `BLSG1Point`      | `Bytes48`      | compressed BLS12-381 G1 point |

*Note*: A subgroup check MUST be performed when deserializing a `BLSG1Point` for use in any of the functions below.

```python
def get_k_commitment(k: BLSFieldElement) -> BLSG1Point:
    return bls.G1_to_bytes48(bls.multiply(bls.G1, k))

def bytes_to_bls_field(b: Bytes32) -> BLSFieldElement:
     """
     Convert bytes to a BLS field scalar. The output is not uniform over the BLS field.
     """
     return int.from_bytes(b, "little") % BLS_MODULUS
```

| Name               | Value                                                                           |
| ------------------ | ------------------------------------------------------------------------------- |
| `BLS_G1_GENERATOR` | `bls.G1_to_bytes48(bls.G1)`                                                     |
| `BLS_MODULUS`      | `52435875175126190479447740508185965837690552500527637822603658699938581184513` |

### Curdleproofs and opening proofs

Note that Curdleproofs (Whisk Shuffle Proofs), the tracker opening proofs and all related data structures and verifier code (along with tests) is specified in [curdleproofs.pie](https://github.com/nalinbhardwaj/curdleproofs.pie/tree/verifier-only) repository.

```
def IsValidWhiskShuffleProof(pre_shuffle_trackers: Sequence[WhiskTracker],
                             post_shuffle_trackers: Sequence[WhiskTracker],
                             shuffle_proof: ByteList[WHISK_MAX_SHUFFLE_PROOF_SIZE]) -> bool:
    """
    Verify `post_shuffle_trackers` is a permutation of `pre_shuffle_trackers`.
    Note: whisk_shuffle_proof_m_commitment is included in whisk_shuffle_proof struct
    Defined in https://github.com/nalinbhardwaj/curdleproofs.pie/tree/verifier-only.
    """
    pass


def IsValidWhiskOpeningProof(tracker: WhiskTracker, k_commitment: BLSG1Point, tracker_proof: ByteList[WHISK_MAX_OPENING_PROOF_SIZE]) -> bool:
    """
    Verify knowledge of `k` such that `tracker.k_r_g == k * tracker.r_g` and `k_commitment == k * BLS_G1_GENERATOR`.
    Defined in https://github.com/nalinbhardwaj/curdleproofs.pie/tree/verifier-only.
    """
    pass
```

## Epoch processing

### `WhiskTracker`

```python
class WhiskTracker(Container):
    r_g: BLSG1Point  # r * G
    k_r_g: BLSG1Point  # k * r * G
```

### `BeaconState`

```python
class BeaconState(bellatrix.BeaconState):
    # ...
    # Whisk
    whisk_candidate_trackers: Vector[WhiskTracker, WHISK_CANDIDATE_TRACKERS_COUNT]  # [New in Whisk]
    whisk_proposer_trackers: Vector[WhiskTracker, WHISK_PROPOSER_TRACKERS_COUNT]  # [New in Whisk]
    whisk_trackers: List[WhiskTracker, VALIDATOR_REGISTRY_LIMIT]  # Whisk tracker (r * G, k * r * G) [New in Whisk]
    whisk_k_commitments: List[BLSG1Point, VALIDATOR_REGISTRY_LIMIT]  # Whisk k commitment k * BLS_G1_GENERATOR [New in Whisk]
```

```python
def select_whisk_proposer_trackers(state: BeaconState, epoch: Epoch) -> None:
    # Select proposer trackers from candidate trackers
    proposer_seed = get_seed(state, epoch - WHISK_PROPOSER_SELECTION_GAP, DOMAIN_WHISK_PROPOSER_SELECTION)
    for i in range(WHISK_PROPOSER_TRACKERS_COUNT):
        index = compute_shuffled_index(uint64(i), uint64(len(state.whisk_candidate_trackers)), proposer_seed)
        state.whisk_proposer_trackers[i] = state.whisk_candidate_trackers[index]


def select_whisk_candidate_trackers(state: BeaconState, epoch: Epoch) -> None:
    # Select candidate trackers from active validator trackers
    active_validator_indices = get_active_validator_indices(state, epoch)
    for i in range(WHISK_CANDIDATE_TRACKERS_COUNT):
        seed = hash(get_seed(state, epoch, DOMAIN_WHISK_CANDIDATE_SELECTION) + uint_to_bytes(i))
        candidate_index = compute_proposer_index(state, active_validator_indices, seed)  # sample by effective balance
        state.whisk_candidate_trackers[i] = state.whisk_trackers[candidate_index]


def process_whisk_updates(state: BeaconState) -> None:
    next_epoch = Epoch(get_current_epoch(state) + 1)
    if next_epoch % WHISK_EPOCHS_PER_SHUFFLING_PHASE == 0:  # select trackers at the start of shuffling phases
        select_whisk_proposer_trackers(state, next_epoch)
        select_whisk_candidate_trackers(state, next_epoch)


def process_epoch(state: BeaconState) -> None:
    bellatrix.process_epoch(state)
    # ...
    process_whisk_updates(state)  # [New in Whisk]
```

## Block processing

### Block header

#### `BeaconBlock`

```python
def process_whisk_opening_proof(state: BeaconState, block: BeaconBlock) -> proNone:
    tracker = state.whisk_proposer_trackers[state.slot % WHISK_PROPOSER_TRACKERS_COUNT]
    k_commitment = state.whisk_k_commitments[block.proposer_index]
    assert IsValidWhiskOpeningProof(tracker, k_commitment, block.body.whisk_opening_proof)


def process_block_header(state: BeaconState, block: BeaconBlock) -> None:
    # bellatrix.process_block_header(state, block)
    # ...
    # [Removed in Whisk] Verify that proposer index is the correct index
    # [Removed in Whisk] assert block.proposer_index == get_beacon_proposer_index(state)
    
    # Verify that the slots match
    assert block.slot == state.slot
    # Verify that the block is newer than latest block header
    assert block.slot > state.latest_block_header.slot
    # # Verify that proposer index is the correct index
    # assert block.proposer_index == get_beacon_proposer_index(state)
    # Verify that the parent matches
    assert block.parent_root == hash_tree_root(state.latest_block_header)
    # Cache current block as the new latest block
    state.latest_block_header = BeaconBlockHeader(
        slot=block.slot,
        proposer_index=block.proposer_index,
        parent_root=block.parent_root,
        state_root=Bytes32(),  # Overwritten in the next process_slot call
        body_root=hash_tree_root(block.body),
    )

    # Verify proposer is not slashed
    proposer = state.validators[block.proposer_index]
    assert not proposer.slashed
    process_whisk_opening_proof(state, block)   # [New in Whisk]
```

### Whisk

#### `BeaconBlockBody`

```python
class BeaconBlockBody(bellatrix.BeaconBlockBody):
    # ...
    # Whisk
    whisk_opening_proof: ByteList[WHISK_MAX_OPENING_PROOF_SIZE]  # [New in Whisk]
    whisk_post_shuffle_trackers: Vector[WhiskTracker, WHISK_VALIDATORS_PER_SHUFFLE]  # [New in Whisk]
    whisk_shuffle_proof: ByteList[WHISK_MAX_SHUFFLE_PROOF_SIZE]  # [New in Whisk]
    whisk_registration_proof: ByteList[WHISK_MAX_OPENING_PROOF_SIZE]  # [New in Whisk]
    whisk_tracker: WhiskTracker  # [New in Whisk]
    whisk_k_commitment: BLSG1Point  # [New in Whisk]
```

```python
def get_shuffle_indices(randao_reveal: BLSSignature) -> Sequence[uint64]:
    """
    Given a `randao_reveal` return the list of indices that got shuffled from the entire candidate set
    """
    indices = []
    for i in WHISK_VALIDATORS_PER_SHUFFLE:
        # XXX ensure we are not suffering from modulo bias
        shuffle_index = uint256(hash(randao_reveal + uint_to_bytes(i))) % WHISK_CANDIDATE_TRACKERS_COUNT
        indices.append(shuffle_index)

    return indices


def whisk_process_shuffled_trackers(state: BeaconState, body: BeaconBlockBody) -> None:
    # Check the shuffle proof
    shuffle_indices = get_shuffle_indices(body.randao_reveal)
    pre_shuffle_trackers = [state.whisk_candidate_trackers[i] for i in shuffle_indices]
    post_shuffle_trackers = body.whisk_post_shuffle_trackers

    shuffle_epoch = get_current_epoch(state) % WHISK_EPOCHS_PER_SHUFFLING_PHASE
    if shuffle_epoch + WHISK_PROPOSER_SELECTION_GAP + 1 >= WHISK_EPOCHS_PER_SHUFFLING_PHASE:
        # Require unchanged trackers during cooldown
        assert pre_shuffle_trackers == post_shuffle_trackers
    else:
        # Require shuffled trackers during shuffle
        assert IsValidWhiskShuffleProof(pre_shuffle_trackers, post_shuffle_trackers, body.whisk_shuffle_proof)

    # Shuffle candidate trackers
    for i, shuffle_index in enumerate(shuffle_indices):
        state.whisk_candidate_trackers[shuffle_index] = post_shuffle_trackers[i]


def is_k_commitment_unique(state: BeaconState, k_commitment: BLSG1Point) -> bool:
    return all([whisk_k_commitment != k_commitment for whisk_k_commitment in state.whisk_k_commitments])


def whisk_process_registration(state: BeaconState, body: BeaconBlockBody) -> None:
    # Overwrite all validator Whisk fields (first Whisk proposal) or just the permutation commitment (next proposals)
    proposer = get_beacon_proposer_index(state)
    if state.whisk_trackers[proposer].r_g == BLS_G1_GENERATOR:  # first Whisk proposal
        assert body.whisk_tracker.r_g != BLS_G1_GENERATOR
        assert is_k_commitment_unique(state, body.whisk_k_commitment)
        assert whisk.IsValidWhiskOpeningProof(body.whisk_tracker, body.whisk_k_commitment, body.whisk_registration_proof)
        state.whisk_trackers[proposer] = body.whisk_tracker
        state.whisk_k_commitments[proposer] = body.whisk_k_commitment
    else:  # next Whisk proposals
        assert body.whisk_registration_proof == WhiskTrackerProof()
        assert body.whisk_tracker == WhiskTracker()
        assert body.whisk_k_commitment == BLSG1Point()


def process_block(state: BeaconState, block: BeaconBlock) -> None:
    bellatrix.process_block(state, block)
    # ...
    whisk_process_shuffled_trackers(state, block.body) # [New in Whisk]
    whisk_process_registration(state, block.body) # [New in Whisk]
```

### Deposits

```python
def get_unique_whisk_k(state: BeaconState, validator_index: ValidatorIndex) -> BLSFieldElement:
    counter = 0
    while True:
        k = BLSFieldElement(bytes_to_bls_field(hash(uint_to_bytes(validator_index) + uint_to_bytes(uint64(counter)))))  # hash `validator_index || counter`
        if is_k_commitment_unique(state, get_k_commitment(k)):
            return k  # unique by trial and error
        counter += 1


def get_initial_tracker(k: BLSFieldElement) -> WhiskTracker:
    # Tracker with r = 1
    WhiskTracker(
        r_g=BLS_G1_GENERATOR,
        k_r_g=get_k_commitment(k)
    )


def process_deposit(state: BeaconState, deposit: Deposit) -> None:
    # Verify the Merkle branch
    assert is_valid_merkle_branch(
        leaf=hash_tree_root(deposit.data),
        branch=deposit.proof,
        depth=DEPOSIT_CONTRACT_TREE_DEPTH + 1,  # Add 1 for the List length mix-in
        index=state.eth1_deposit_index,
        root=state.eth1_data.deposit_root,
    )

    # Deposits must be processed in order
    state.eth1_deposit_index += 1

    pubkey = deposit.data.pubkey
    amount = deposit.data.amount
    validator_pubkeys = [validator.pubkey for validator in state.validators]
    if pubkey not in validator_pubkeys:
        # Verify the deposit signature (proof of possession) which is not checked by the deposit contract
        deposit_message = DepositMessage(
            pubkey=deposit.data.pubkey,
            withdrawal_credentials=deposit.data.withdrawal_credentials,
            amount=deposit.data.amount,
        )
        domain = compute_domain(DOMAIN_DEPOSIT)  # Fork-agnostic domain since deposits are valid across forks
        signing_root = compute_signing_root(deposit_message, domain)
        # Initialize validator if the deposit signature is valid
        if bls.Verify(pubkey, signing_root, deposit.data.signature):
            state.validators.append(get_validator_from_deposit(state, deposit))
            state.balances.append(amount)
            state.previous_epoch_participation.append(ParticipationFlags(0b0000_0000))
            state.current_epoch_participation.append(ParticipationFlags(0b0000_0000))
            state.inactivity_scores.append(uint64(0))
            # [New in Whisk]
            k = get_unique_whisk_k(state, ValidatorIndex(len(state.validators) - 1))
            state.whisk_trackers.append(get_initial_tracker(k))
            state.whisk_k_commitments.append(get_k_commitment(k))

    else:
        # Increase balance by deposit amount
        index = ValidatorIndex(validator_pubkeys.index(pubkey))
        increase_balance(state, index, amount)
```

### `get_beacon_proposer_index`

```python
def get_beacon_proposer_index(state: BeaconState) -> ValidatorIndex:
    """
    Return the beacon proposer index at the current slot.
    """
    return state.latest_block_header.proposer_index
```
