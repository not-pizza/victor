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

async function storeEmbedding(
  embedInput: string,
  openaiApiKey: string,
  shouldStore = true,
  shouldSearch = true,
) {
  const embedResponse = await fetchEmbedding(embedInput, openaiApiKey);

  if (!embedResponse.success) {
    return;
  }

  const root = await navigator.storage.getDirectory();

  const embedding = new Float64Array(embedResponse.data.data[0].embedding);

  if (shouldSearch) {
    const result = await victor.find_nearest_neighbor(root, embedding);
    console.log(result);
  }

  if (shouldStore) {
    await victor.write_embedding(root, embedInput, embedding);
  }
}

async function storeFlatlandEmbedding(
  embedding: number[],
  embeddingText: string,
) {
  const root = await navigator.storage.getDirectory();
  await victor.write_embedding(
    root,
    embeddingText,
    new Float64Array(embedding),
  );
}

async function onSubmitEmbedding() {
  const openaiApiKey = (
    document.querySelector('input[name="openai"]') as HTMLInputElement
  ).value;
  if (openaiApiKey !== '' && openaiApiKey !== undefined) {
    localStorage.setItem('openaiApiKey', openaiApiKey);
  }
  const embedInput = (
    document.querySelector('textarea[name="embedInput"]') as HTMLInputElement
  ).value;

  const searchInput = (
    document.querySelector('textarea[name="searchInput"]') as HTMLInputElement
  ).value;

  await storeEmbedding(
    !!embedInput ? embedInput : searchInput,
    openaiApiKey,
    !!embedInput,
    !!searchInput,
  );
}

async function embedFlatlands() {
  for (let i = 0; i < flatland.paragraphs.length; i++) {
    const paragraph = flatland.paragraphs[i];
    const embedding = flatland.embeddings[i];
    if (embedding && paragraph.length > 25) {
      console.log(flatland.paragraphs.length - i);
      await storeFlatlandEmbedding(embedding[0] as number[], paragraph);
    }
    console.log(flatland.paragraphs.length - i);
  }
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
      await victor.clear_db(root);
    } catch (e) {
      console.log('could not clear:', e);
    }
  }
}

// Expose the functions to the global window object so they're accessible from HTML
(window as any).onSubmitEmbedding = onSubmitEmbedding;
(window as any).clearDb = clearDb;
(window as any).embedFlatlands = embedFlatlands;
