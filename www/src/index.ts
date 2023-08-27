import * as victor from 'victor';
import flatland from './flatland.json';

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

  let current_nearest = await victor.find_nearest_neighbor(root, embedding);
  console.log('nearest:', current_nearest);

  await victor.write_embedding(root, embedding, embedInput);
}

async function onSubmitStoreEmbedding() {
  const openaiApiKey = (
    document.querySelector('input[name="openai"]') as HTMLInputElement
  ).value;
  if (openaiApiKey !== '' && openaiApiKey !== undefined) {
    localStorage.setItem('openaiApiKey', openaiApiKey);
  }
  const embedInput = (
    document.querySelector('input[name="embedInput"]') as HTMLInputElement
  ).value;

  await storeEmbedding(embedInput, openaiApiKey);
}

async function embedFlatlands() {
  flatland.paragraphs.forEach(async (paragraph: string) => {
    await storeEmbedding(paragraph, localStorage.getItem('openaiApiKey'));
  });
}

function restoreOpenaiApiKey() {
  console.log('restoring openai api key');
  const openaiApiKey = localStorage.getItem('openaiApiKey');
  if (openaiApiKey !== '' && openaiApiKey !== undefined) {
    (document.querySelector('input[name="openai"]') as HTMLInputElement).value =
      openaiApiKey;
  }
}

restoreOpenaiApiKey();

async function clearDb() {
  console.log('clearing db');
  const root = await navigator.storage.getDirectory();
  if (root) {
    try {
      await root.removeEntry('victor.bin');
    } catch (e) {
      console.log('could not clear:', e);
    }
  }
}

// Expose the functions to the global window object so they're accessible from HTML
(window as any).onSubmitStoreEmbedding = onSubmitStoreEmbedding;
(window as any).clearDb = clearDb;
(window as any).embedFlatlands = embedFlatlands;
