export type EmbeddingResponse = {
  data: Array<{
    embedding: number[];
    index: number;
    object: 'embedding';
  }>;
  model: string;
  object: 'list';
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
};
