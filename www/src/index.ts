import * as victor from 'victor';
import { EmbeddingResponse } from '../types/openai';

type Success<T> = {
  success: true;
  data: T;
};

type Failure = {
  success: false;
  error: string;
};

type Result<T> = Success<T> | Failure;

async function fetchEmbedding(
  embedInput: string,
  openaiApiKey: string,
): Promise<Result<EmbeddingResponse>> {
  try {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${openaiApiKey}`,
      },
      body: JSON.stringify({
        input: embedInput,
        model: 'text-embedding-ada-002',
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('Error:', error);
    return { success: false, error: error.message };
  }
}

async function storeEmbedding(embedInput: string, openaiApiKey: string) {
  const embedResponse = await fetchEmbedding(embedInput, openaiApiKey);

  if (!embedResponse.success) {
    return;
  }

  console.log(embedResponse.data);

  const root = await navigator.storage.getDirectory();

  const embedding = new Float64Array(embedResponse.data.data[0].embedding);

  await victor.embed(root, embedding);

  const fileHandle = await root.getFileHandle('victor.bin', { create: false });
  const file = await fileHandle.getFile();
  const contents = await file.text();
  console.log(contents);
}

async function onSubmit() {
  const openaiApiKey = (
    document.querySelector('input[name="openai"]') as HTMLInputElement
  ).value;
  const embedInput = (
    document.querySelector('input[name="embedinput"]') as HTMLInputElement
  ).value;

  await storeEmbedding(embedInput, openaiApiKey);
}

// Expose the function to the global window object so it's accessible from HTML
(window as any).onSubmit = onSubmit;
