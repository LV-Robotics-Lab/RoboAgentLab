// Use the reviewed catalog pin so provider listing changes cannot change CI input.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { hydrateModelCatalogBytes } from "./update-model-catalog-pin.mjs";

const root = process.cwd();
const { revision } = JSON.parse(readFileSync("nix/model-catalog.json", "utf8"));
if (!/^sha256-[0-9a-f]{64}$/.test(revision)) throw new Error("Invalid model catalog revision");
const response = await fetch(`https://pi.dev/api/models/revisions/${revision}?types=chat,image,classifier`);
if (!response.ok) throw new Error(`Catalog download failed: HTTP ${response.status}`);
const bytes = Buffer.from(await response.arrayBuffer());
if (`sha256-${createHash("sha256").update(bytes).digest("hex")}` !== revision)
	throw new Error("Model catalog checksum mismatch");
hydrateModelCatalogBytes(root, bytes);
console.log(`Hydrated pinned model catalog ${revision}`);
