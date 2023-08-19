import * as victor from 'victor';

async function create() {
  const root = await navigator.storage.getDirectory();
  const fileHandle = await root.getFileHandle('victor.txt', { create: true });
  const writable = await fileHandle.createWritable();

  await writable.write(victor.init_db_contents());
  await writable.close();

  const file = await fileHandle.getFile();
  const contents = await file.text();
  console.log(contents);
}
create();
