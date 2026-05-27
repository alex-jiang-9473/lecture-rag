export interface ChunkContext {
  source: string;
  page: number;
  chunk_index: number;
  score: number;
  text: string;
}

export interface AskRequest {
  question: string;
  top_k?: number;
}

export interface AskResponse {
  answer: string;
  sources: string[];
  context: ChunkContext[];
}

export interface HealthResponse {
  status: string;
}
