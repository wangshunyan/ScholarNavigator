import type { RankedPaper } from "@/types/api";

export const TOP20_DELIVERY_VIEW_VERSION = "frontend_top20_delivery_v1";

/**
 * Stable React identity for one delivered paper.
 *
 * The API derives this value from the unified paper identity contract. Rank and
 * array position are deliberately excluded so pagination or an exact-tie order
 * change cannot cause React to reuse a card for a different paper.
 */
export function top20PaperKey(paper: RankedPaper): string {
  return paper.result_identity;
}
