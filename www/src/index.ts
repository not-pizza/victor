import * as victor from 'victor';

async function create() {
  const root = await navigator.storage.getDirectory();
  const fileHandle = await root.getFileHandle('victor.txt', { create: true });

  await victor.init_db_contents(fileHandle);

  const file = await fileHandle.getFile();
  const contents = await file.text();
  console.log(contents);
}
create();
