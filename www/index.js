import * as victor from "victor";

async function example() {
    const root = await navigator.storage.getDirectory();
    victor.greet(root);
}
example()