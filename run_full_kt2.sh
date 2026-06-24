
#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  run_full_kt2.sh — kompletni KT2 pipeline za Person 1 (Luka)
#
#  Pokreće:
#    1) Brzi smoke test (~45 min) — provjera da sve radi
#    2) Puni run (~4-5 sati) — dynamic pruning + kvantizacija na V2 modelu
#
#  Sve logove i modele stavlja u zasebne foldere s timestampom da se
#  ne zgaze. Ako bilo koji korak padne, skripta se zaustavlja.
#
#  Pokretanje:
#      chmod +x run_full_kt2.sh
#      ./run_full_kt2.sh
#
#  Ili u pozadini (preživi zatvaranje terminala):
#      nohup ./run_full_kt2.sh > run_kt2.out 2>&1 &
#
#  Pratiti napredak (u drugom terminalu):
#      tail -f run_kt2.out
# ════════════════════════════════════════════════════════════════════════════

set -e   # ako bilo koja komanda padne, zaustavi cijelu skriptu
set -u   # padaj ako se referenciram na nedefiniranu varijablu

# ── Postavke ────────────────────────────────────────────────────────────────
DATA_DIR="./fruits-360-100x100"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
RESULTS_DIR="results_kt2_${TIMESTAMP}"

# ── Logging helper ─────────────────────────────────────────────────────────
log_step() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(date +'%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
}

# ── Provjera prije pokretanja ──────────────────────────────────────────────
log_step "PROVJERA OKRUŽENJA"

# Provjeri da dataset postoji
if [ ! -d "$DATA_DIR" ]; then
    echo "  ✗ GREŠKA: Dataset nije pronađen u $DATA_DIR"
    echo "  Skidaj Fruits-360 i smjesti ga ovdje, ili promijeni DATA_DIR u skripti."
    exit 1
fi
echo "  ✓ Dataset pronađen: $DATA_DIR"

# Provjeri sve potrebne Python skripte
required_files=(
    "person1_model.py"
    "person1_model_v2.py"
    "person2_train.py"
    "person3_dynamic_pruning.py"
    "person4_quantization_v2.py"
    "logger.py"
)
for f in "${required_files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  ✗ GREŠKA: Nedostaje datoteka: $f"
        exit 1
    fi
    echo "  ✓ $f"
done

# Provjeri Python i NumPy
if ! command -v python &> /dev/null; then
    echo "  ✗ GREŠKA: python nije instaliran (ili nije u PATH)"
    exit 1
fi
echo "  ✓ Python: $(python --version)"

if ! python -c "import numpy" 2>/dev/null; then
    echo "  ✗ GREŠKA: NumPy nije instaliran"
    exit 1
fi
echo "  ✓ NumPy: $(python -c 'import numpy; print(numpy.__version__)')"

# Stvori folder za rezultate
mkdir -p "$RESULTS_DIR"
echo "  ✓ Folder za rezultate: $RESULTS_DIR"

# Spremi početne timestampove
START_TIME=$(date +%s)

# ════════════════════════════════════════════════════════════════════════════
#  KORAK 1 — BRZI SMOKE TEST
#  ~30-45 min, max_per_class=30, epochs=5
# ════════════════════════════════════════════════════════════════════════════

log_step "KORAK 1 — BRZI SMOKE TEST (~30-45 min)"

# 1a) Dynamic pruning sa malim datasetom
echo "  [1a] Pokretanje dynamic pruning (test)..."
python person3_dynamic_pruning.py \
    --data_dir "$DATA_DIR" \
    --epochs 5 \
    --max_per_class 30 \
    --amount 0.5 \
    --save_path "models/test_pruned" \
    --mask_path "models/test_pruned_masks" \
    2>&1 | tee "$RESULTS_DIR/01_pruning_test.log"

# Provjera da je model spremljen
if [ ! -f "models/test_pruned.npz" ]; then
    echo ""
    echo "  ✗ GREŠKA: Dynamic pruning nije proizveo model (models/test_pruned.npz)"
    echo "  Pogledaj log: $RESULTS_DIR/01_pruning_test.log"
    exit 1
fi
echo "  ✓ Brzi test pruninga prošao"

# 1b) Kvantizacija nad pruned modelom
echo ""
echo "  [1b] Pokretanje kvantizacije (test)..."
python person4_quantization_v2.py \
    --data_dir "$DATA_DIR" \
    --model_path "models/test_pruned" \
    --save_path "models/test_final" \
    --max_per_class 30 \
    --strategy asymmetric \
    2>&1 | tee "$RESULTS_DIR/02_quantization_test.log"

if [ ! -f "models/test_final.npz" ]; then
    echo ""
    echo "  ✗ GREŠKA: Kvantizacija nije proizvela model"
    echo "  Pogledaj log: $RESULTS_DIR/02_quantization_test.log"
    exit 1
fi
echo "  ✓ Brzi test kvantizacije prošao"

SMOKE_END=$(date +%s)
SMOKE_DURATION=$(( (SMOKE_END - START_TIME) / 60 ))
log_step "✓ BRZI TEST ZAVRŠEN ZA $SMOKE_DURATION min"

# ════════════════════════════════════════════════════════════════════════════
#  KORAK 2 — PUNI RUN
#  ~4-5 sati, max_per_class=100, epochs=15
# ════════════════════════════════════════════════════════════════════════════

log_step "KORAK 2 — PUNI RUN (~4-5 sati)"
echo "  Brzi test je uspio, krećem s pravim runom."
echo "  Slobodno zatvori laptop ekran (ali ne idi u sleep)."
echo ""

# 2a) Dynamic pruning na cijelom datasetu
echo "  [2a] Pokretanje pravog dynamic pruninga..."
python person3_dynamic_pruning.py \
    --data_dir "$DATA_DIR" \
    --epochs 15 \
    --max_per_class 100 \
    --amount 0.5 \
    --strategy global \
    --prune_start_epoch 2 \
    --prune_frequency 10 \
    --save_path "models/v2_pruned_final" \
    --mask_path "models/v2_pruned_masks_final" \
    2>&1 | tee "$RESULTS_DIR/03_pruning_full.log"

if [ ! -f "models/v2_pruned_final.npz" ]; then
    echo "  ✗ GREŠKA: Pravi dynamic pruning nije proizveo model"
    exit 1
fi

PRUNING_END=$(date +%s)
PRUNING_DURATION=$(( (PRUNING_END - SMOKE_END) / 60 ))
echo "  ✓ Pravi pruning gotov za $PRUNING_DURATION min"

# 2b) Kvantizacija
echo ""
echo "  [2b] Pokretanje prave kvantizacije..."
python person4_quantization_v2.py \
    --data_dir "$DATA_DIR" \
    --model_path "models/v2_pruned_final" \
    --save_path "models/v2_final_quantized" \
    --max_per_class 100 \
    --strategy asymmetric \
    2>&1 | tee "$RESULTS_DIR/04_quantization_full.log"

if [ ! -f "models/v2_final_quantized.npz" ]; then
    echo "  ✗ GREŠKA: Prava kvantizacija nije proizvela model"
    exit 1
fi

# ════════════════════════════════════════════════════════════════════════════
#  KORAK 3 — SAŽETAK
# ════════════════════════════════════════════════════════════════════════════

END_TIME=$(date +%s)
TOTAL_DURATION=$(( (END_TIME - START_TIME) / 60 ))

log_step "✓ SVE ZAVRŠENO — UKUPNO $TOTAL_DURATION min"

# Stvori sažetak
SUMMARY="$RESULTS_DIR/SUMMARY.txt"
{
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  KT2 PIPELINE SAŽETAK"
    echo "  $(date +'%Y-%m-%d %H:%M')"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    echo "Brzi test trajao: $SMOKE_DURATION min"
    echo "Pravi pruning trajao: $PRUNING_DURATION min"
    echo "Ukupno: $TOTAL_DURATION min"
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo "  FINALNI MODELI (u models/ folderu):"
    echo "─────────────────────────────────────────────────────────────────"
    echo "  v2_pruned_final.npz       ← V2 + Dynamic Pruning"
    echo "  v2_pruned_final_bn.npz    ← BatchNorm parametri"
    echo "  v2_pruned_masks_final.npz ← Pruning maske"
    echo "  v2_final_quantized.npz    ← V2 + Pruning + INT8 kvantizacija"
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo "  LOGOVI U FOLDERU $RESULTS_DIR:"
    echo "─────────────────────────────────────────────────────────────────"
    echo "  01_pruning_test.log         ← brzi test pruning"
    echo "  02_quantization_test.log    ← brzi test kvantizacija"
    echo "  03_pruning_full.log         ← pravi pruning"
    echo "  04_quantization_full.log    ← prava kvantizacija"
    echo ""
    echo "  Dodatno u logs/ folderu su točniji per-script timestamped logovi."
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo "  REZULTATI iz pravog runa (zadnje epohe):"
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    echo "  Dynamic pruning (zadnji red iz 03_pruning_full.log):"
    grep -E "Epoch \[ *15/15\]|Best test accuracy|Final sparsity" \
        "$RESULTS_DIR/03_pruning_full.log" | tail -5 | sed 's/^/    /'
    echo ""
    echo "  Kvantizacija (iz 04_quantization_full.log):"
    grep -E "Accuracy|Theoretical Compression|Speedup|INT8 Size" \
        "$RESULTS_DIR/04_quantization_full.log" | tail -10 | sed 's/^/    /'
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
} | tee "$SUMMARY"

echo ""
echo "  Cijeli sažetak: $SUMMARY"
echo ""

