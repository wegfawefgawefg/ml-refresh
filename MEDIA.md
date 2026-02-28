# Media Manifest

This project uses large local media files for experiments in `src/sq_nextframe_1`.
These assets are intentionally excluded from git via `.gitignore` (for example `*.mp4`, `*.png`).

## Source videos (not committed)

- `src/sq_nextframe_1/videos/secret_life_of_pets/The.Secret.Life.Of.Pets.2016.1080p.BluRay.x264-[YTS.AG].mp4`
  - Size: 1,408,762,135 bytes (~1.31 GiB)
  - SHA-256: `7e9a2925616819963eeae0649b200f3c20b521192e915dd00ae51df34ae382b9`
  - Notes: filename indicates a YTS release.
- `src/sq_nextframe_1/videos/bbb_60fps.mp4`
  - Size: 339,489,735 bytes (~323.8 MiB)
  - SHA-256: `658cb0019af04f7016b9686a6329e9120f97cb7d0cb67ab5fa0af6dd4f519e40`

## Derived media directories (not committed)

- `src/sq_nextframe_1/videos/secret_life_of_pets/128_128` (~4.3 GiB)
- `src/sq_nextframe_1/videos/secret_life_of_pets/28_28` (~586 MiB)
- `src/sq_nextframe_1/generated_frames` (~27 MiB)

## Reacquire checklist

1. Re-obtain source videos from your original source/library.
2. Verify identity with:
   - `sha256sum <file>`
3. Place files back under:
   - `src/sq_nextframe_1/videos/...`
4. Re-run data prep/train/generate scripts in `src/sq_nextframe_1` as needed.

## Licensing note

Make sure you have rights to use any video assets. Keep copyrighted media out of the repository.
