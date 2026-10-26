# EEG BCI Signal Conditioning App — MP3/MP4 exports

## Run in VS Code
1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. In the VS Code terminal, run:
   `python -m pip install -r requirements.txt`
4. Run:
   `python -m streamlit run app.py`

## Export formats
- PNG: static plots.
- CSV: processed sample data.
- MP3: educational sonification. EEG amplitude modulates an audible 440 Hz tone; this is not a literal recording of EEG frequencies.
- MP4: animated visualization of raw EEG, processed EEG, theta, alpha, and beta traces, with the sonification audio included as a soundtrack.

MP3 generation uses `lameenc`; MP4 generation uses `imageio-ffmpeg`. The app catches missing-export dependencies and displays an error instead of stopping the whole page.
