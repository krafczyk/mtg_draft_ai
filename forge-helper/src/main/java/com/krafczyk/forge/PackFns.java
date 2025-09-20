package com.krafczyk.forge;
import java.util.*;
import forge.item.PaperCard;

public class PackFns {
  public static final class Csr {
    public final int[] indptr;
    public final int[] indices;
    public final byte[] data;  // 0/1 counts
    public final int ncols;
    public Csr(int[] indptr, int[] indices, byte[] data, int ncols) {
      this.indptr = indptr; this.indices = indices; this.data = data; this.ncols = ncols;
    }
  }

  private static int requireIndex(String name, Map<String,Integer> index) {
    Integer j = index.get(name);
    if (j == null) {
      throw new IllegalArgumentException("Card not in supplied index: " + name);
    }
    return j.intValue();
  }

  public static Csr packsToCsr(
      List<? extends List<? extends PaperCard>> packs,
      Map<String,Integer> index, int ncols) {

    final int B = packs.size();

    // First pass: per-row counts (col -> count), and track nnz per row
    @SuppressWarnings("unchecked")
    HashMap<Integer,Integer>[] rowCounts = new HashMap[B];
    int totalNnz = 0;

    for (int i = 0; i < B; i++) {
      HashMap<Integer,Integer> counts = new HashMap<>(16);
      for (PaperCard pc : packs.get(i)) {
        int col = requireIndex(pc.getName(), index);  // adjust accessor if needed
        counts.merge(col, 1, Integer::sum);
      }
      rowCounts[i] = counts;
      totalNnz += counts.size();
    }

    // Allocate CSR arrays
    int[] indptr  = new int[B + 1];
    int[] indices = new int[totalNnz];
    byte[] data   = new byte[totalNnz];

    // Second pass: write rows (sorted by column for canonical CSR)
    int pos = 0;
    for (int i = 0; i < B; i++) {
      indptr[i] = pos;

      HashMap<Integer,Integer> counts = rowCounts[i];
      // sort column indices to keep CSR canonical
      int[] cols = counts.keySet().stream().mapToInt(Integer::intValue).sorted().toArray();

      for (int c : cols) {
        int v = counts.get(c);
        indices[pos] = c;
        data[pos] = (byte) v;  // if v can exceed 127, switch data[] to short[] or int[]
        pos++;
      }
    }
    indptr[B] = pos;

    return new Csr(indptr, indices, data, ncols);
  }

  public static byte[][] packsToDense(List<? extends List<? extends forge.item.PaperCard>> packs,
                                      java.util.Map<String,Integer> index, int ncols) {
    final int n = packs.size();
    byte[][] M = new byte[n][ncols];  // zero-initialized

    for (int i = 0; i < n; i++) {
      for (PaperCard pc : packs.get(i)) {
        int j = requireIndex(pc.getName(), index);  // adjust accessor if needed
        M[i][j] = (byte) (M[i][j] + 1);            // increment count
      }
    }
    return M;
  }
}
