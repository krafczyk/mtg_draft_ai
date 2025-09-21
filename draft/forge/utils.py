import numpy as np
import pandas as pd


def get_edition(edition_code: str):
    import jpype.imports
    from forge.model import FModel
    return FModel.getMagicDb().getCardEdition(edition_code)


def get_draftable_cards_from_edition(edition):
    import jpype.imports
    from forge.item.generation import BoosterGenerator

    cards = set()
    sealed_template = edition.getBoosterTemplate()
    for slot_name, _ in sealed_template.getSlots():
        slot = sealed_template.getNamedSlots().get(slot_name.replace('+', ''))
        print(slot_name)
        for _, sheet in slot.getSlotSegments().items():
            ps = BoosterGenerator.makeSheet(sheet, [])
            for pc in ps.toFlatList():
                cards.add(pc.getName())
    cards = list(cards)
    cards.sort()
    return cards


def java_pack_list_to_sparse_df(java_packs, cards):
    import jpype
    from scipy import sparse
    # Build vocab
    card_to_idx = {card: idx for idx, card in enumerate(cards)}
    HashMap = jpype.JClass("java.util.HashMap")
    j_index = HashMap()
    for k, v in card_to_idx.items():
        j_index.put(k, jpype.types.JInt(v))

    # Build sparse matrix
    from com.krafczyk.forge import PackFns
    csr_res = PackFns.packsToCsr(
        java_packs,
        j_index,
        jpype.types.JInt(len(cards)))
    indptr = np.asarray(csr_res.indptr, dtype=np.int32, copy=True)
    indices = np.asarray(csr_res.indices, dtype=np.int32, copy=True)
    data_i8 = np.asarray(csr_res.data, dtype=np.int8, copy=True)
    data_i32 = data_i8.astype(np.int32, copy=False)   # widen
    A = sparse.csr_matrix((data_i32, indices, indptr), shape=(indptr.size-1, len(cards)))
    #A.sum_duplicates()
    return pd.DataFrame.sparse.from_spmatrix(A, columns=cards)


def java_pack_list_to_dense_df(java_packs, cards):
    import jpype
    # Build vocab
    card_to_idx = {card: idx for idx, card in enumerate(cards)}
    HashMap = jpype.JClass("java.util.HashMap")
    j_index = HashMap()
    for k, v in card_to_idx.items():
        j_index.put(k, jpype.types.JInt(v))

    from com.krafczyk.forge import PackFns
    dense_java = PackFns.packsToDense(java_packs, j_index, jpype.types.JInt(len(cards)))

    # Convert to NumPy and DataFrame
    dense = np.asarray(dense_java, dtype=np.int8, copy=True).view(np.uint8)  # byte[][] -> uint8
    return pd.DataFrame(dense, columns=cards)
