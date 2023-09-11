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

async function fetchEmbedding(embedInput: string, openaiApiKey: string): Promise<Result<EmbeddingResponse>> {
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

async function getRootDirectory(): Promise<FileSystemDirectoryHandle> {
  return await navigator.storage.getDirectory();
}

async function storeEmbedding(embedInput: string, openaiApiKey: string, tags: string[]) {
  const embedResponse = await fetchEmbedding(embedInput, openaiApiKey);

  if (!embedResponse.success) {
    return;
  }

  const root = await getRootDirectory();
  const embedding = new Float64Array(embedResponse.data.data[0].embedding);
  await victor.write_embedding(root, embedInput, embedding, tags);
}

async function searchEmbedding(embedInput: string, openaiApiKey: string, tags: string[]) {
  const embedResponse = await fetchEmbedding(embedInput, openaiApiKey);

  if (!embedResponse.success) {
    return;
  }

  const root = await getRootDirectory();
  const embedding = new Float64Array(embedResponse.data.data[0].embedding);
  const result = await victor.find_nearest_neighbor(root, embedding, tags);
  console.log(result);
}

function getFormValues(): { openaiApiKey: string; inputText: string; tags: string[] } {
  const openaiApiKey = (document.getElementById('openai') as HTMLInputElement).value;
  const inputText = (document.getElementById('inputText') as HTMLInputElement).value;
  const tags = (document.getElementById('tags') as HTMLInputElement).value.split(',').map(tag => tag.trim()).filter(tag => tag !== '');

  if (openaiApiKey) {
    localStorage.setItem('openaiApiKey', openaiApiKey);
  }

  return { openaiApiKey, inputText, tags };
}

async function handleEmbed() {
  const { openaiApiKey, inputText, tags } = getFormValues();

  if (inputText) {
    await storeEmbedding(inputText, openaiApiKey, tags);
  }
}

async function handleSearch() {
  const { openaiApiKey, inputText, tags } = getFormValues();

  if (inputText) {
    await searchEmbedding(inputText, openaiApiKey, tags);
  }
}

async function storeFlatlandEmbedding(embedding: number[], embeddingText: string) {
  const root = await navigator.storage.getDirectory();
  await victor.write_embedding(
    root,
    embeddingText,
    new Float64Array(embedding),
    ['flatland'],
  );
}

async function embedAllFlatlands() {
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

async function clearDatabase() {
  console.log('clearing db');
  const root = await navigator.storage.getDirectory();
  console.log('root: ', root);
  if (root) {
    try {
      await victor.clear_db(root);
    } catch (e) {
      console.log('could not clear:', e);
    }
  }
}

// Exposing the functions to the global window object
(window as any).handleEmbed = handleEmbed;
(window as any).handleSearch = handleSearch;
(window as any).clearDatabase = clearDatabase;
(window as any).embedAllFlatlands = embedAllFlatlands;