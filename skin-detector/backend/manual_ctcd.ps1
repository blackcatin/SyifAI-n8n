# Pengujian MANUAL CT/CD SyifAI (pembanding n8n) - revisi poin 5 & 7  (PowerShell)
# Pakai:  . .\manual_ctcd.ps1     (perhatikan titik + spasi di depan)

# ====== KONFIGURASI ======
$script:API          = "http://localhost:8000"
$script:HF_TRAIN     = "https://blackcatin-syifai-trainer.hf.space"   # URL sudah dikonfirmasi (Space berstatus Running)
$script:HF_TRAIN_PATH= "/train"                                       # sudah dikonfirmasi dari train.py di HF Space
$script:MIN_DATA     = 5
# Hitung data baru lewat Python + database.py/models.py milikmu (tanpa psql). Jalankan script dari folder backend.
# Cari Python yang dipakai FastAPI: venv aktif, lalu venv/.venv, lalu py, lalu python. Atau isi manual SETELAH memuat script: $script:PYEXE = "D:\...\python.exe"
$script:PYEXE = $null
if ($env:VIRTUAL_ENV -and (Test-Path "$env:VIRTUAL_ENV\Scripts\python.exe")) { $script:PYEXE = "$env:VIRTUAL_ENV\Scripts\python.exe" }
if (-not $script:PYEXE) {
  foreach ($c in @(".\venv\Scripts\python.exe", ".\.venv\Scripts\python.exe", "..\venv\Scripts\python.exe", "..\.venv\Scripts\python.exe")) {
    if (Test-Path $c) { $script:PYEXE = (Resolve-Path $c).Path; break }
  }
}
if (-not $script:PYEXE) { foreach ($c in @("py","python")) { if (Get-Command $c -ErrorAction SilentlyContinue) { $script:PYEXE = $c; break } } }
if ($script:PYEXE) { Write-Host "Python: $script:PYEXE" } else { Write-Host "Python TIDAK ditemukan. Isi manual: `$script:PYEXE = 'D:\...\python.exe'" -ForegroundColor Red }
$script:DB_CMD = { & $script:PYEXE .\db_helper.py count }   # butuh db_helper.py di folder backend

$script:LOG   = "manual_run_$(Get-Date -Format yyyyMMdd_HHmmss).csv"
$script:STEPS = 0
$script:ROWS  = @()
$script:ERRS  = 0

function Step($t) { $script:STEPS++; Write-Host "  [langkah manual #$($script:STEPS)] $t" -ForegroundColor Yellow }
function Pause-Manual($t) { Read-Host "  $t (Enter bila selesai)" | Out-Null }

function Run-Stage($name, [scriptblock]$block) {
  Write-Host "`n=== $name ===" -ForegroundColor Cyan
  $s0 = $script:STEPS; $sw = [Diagnostics.Stopwatch]::StartNew()
  $ok = $false
  try { $res = @(& $block); $ok = ($res.Count -gt 0 -and $res[-1] -eq $true) }
  catch { Write-Host "  GAGAL: $($_.Exception.Message)" -ForegroundColor Red; $script:ERRS++ }
  $sw.Stop()
  $st = if ($ok) { "OK" } else { "GAGAL/BERHENTI" }
  $script:ROWS += [pscustomobject]@{ tahap=$name; durasi_detik=[math]::Round($sw.Elapsed.TotalSeconds,2); langkah_manual=($script:STEPS-$s0); status=$st }
  return $ok
}

function Show-Summary($totalSec) {
  $train = ($script:ROWS | Where-Object tahap -eq "3_BuildModel").durasi_detik
  $script:ROWS += [pscustomobject]@{ tahap="TOTAL"; durasi_detik=[math]::Round($totalSec,2); langkah_manual=$script:STEPS; status="error=$($script:ERRS)" }
  $script:ROWS += [pscustomobject]@{ tahap="WAKTU_OPERATOR (total - training)"; durasi_detik=[math]::Round($totalSec-[double]$train,2); langkah_manual=$script:STEPS; status="" }
  $script:ROWS | Export-Csv $script:LOG -NoTypeInformation
  Write-Host "`n--- HASIL ---" -ForegroundColor Green
  $script:ROWS | Format-Table -AutoSize
  Write-Host "Tersimpan: $script:LOG"
}

# ====== PERSIAPAN DATA UJI (tiap siklus butuh >=5 data baru) ======
# Pakai: seed_data "D:\path\folder_gambar"
function seed_data($folder, $n = 5) {
  if (-not (Test-Path $folder)) { Write-Host "Folder tidak ada: $folder" -ForegroundColor Red; return }
  # Kirim gambar acak sampai terkumpul $n data VALID (lolos validator kualitas). Gambar tidak valid dilewati.
  $files = Get-ChildItem $folder -Include *.jpg,*.jpeg,*.png -Recurse | Get-Random -Count 60
  $ok = 0; $tried = 0
  foreach ($f in $files) {
    if ($ok -ge $n) { break }
    $tried++
    $r = curl.exe -s -X POST "$($script:API)/predict/" -F "file=@$($f.FullName)" | ConvertFrom-Json
    $valid = [bool]$r.quality.is_valid_for_training
    if ($valid) { $ok++ }
    Write-Host ("  {0}  valid={1}" -f $f.Name, $valid)
  }
  Write-Host "`nValid terkumpul: $ok dari $tried gambar dikirim." -ForegroundColor Cyan
  if ($ok -lt $n) { Write-Host "Kurang dari $n. Coba folder lain (mis. .\uploads\raw)." -ForegroundColor Red }
  Write-Host "Data baru (belum dipakai) sekarang: $(& $script:DB_CMD | Select-Object -Last 1)"
}


# Backlog data lama (mis. sisa uji k6) disisihkan supaya tiap siklus hanya memakai N data baru, seperti uji n8n.
# Pakai SEBELUM tiap run_cycle:  prepare_data 5      Setelah semua uji selesai:  restore_data
function prepare_data($n = 5) { & $script:PYEXE .\db_helper.py prepare $n | ForEach-Object { Write-Host $_ }; Write-Host "(angka terakhir = data baru siap)" }
function restore_data { & $script:PYEXE .\db_helper.py restore | ForEach-Object { Write-Host $_ }; Write-Host "(angka terakhir = data baru setelah restore)" }

# ====== 1. INGESTION ======
function ingestion {
  Step "buka terminal & koneksi DB, tulis query hitung data baru"
  $n = [int]((& $script:DB_CMD) | Select-Object -Last 1)
  Write-Host "  Data baru: $n (minimum $script:MIN_DATA)"
  Step "bandingkan jumlah data dengan batas minimum"
  if ($n -lt $script:MIN_DATA) { Write-Host "  -> DATA BELUM CUKUP (skipped)" -ForegroundColor Red; return $false }
  Write-Host "  -> data cukup"; return $true
}

# ====== 2. PREPARATION ======
function preparation {
  Step "request model aktif"
  $a = Invoke-RestMethod "$script:API/active-model-filename"; $a | Out-Host
  $script:PREV_FILE = $a.filename
  # Nama model uji dibuat unik (model_manual_<waktu>) supaya TIDAK menimpa model_v5.pth yang sudah ada di HF
  # dan tidak mengganggu penomoran versi n8n (/latest-version hanya menghitung file berawalan "model_v").
  $script:NEW_VER = "model_manual_" + (Get-Date -Format "MMddHHmm")
  Step "tentukan nama versi model baru dari versi aktif ($($a.version)) -> $script:NEW_VER"
  Step "request arsip & siapkan data per kelas"
  $script:ARCH = Invoke-RestMethod -Method Post "$script:API/archive-retraining-data"
  Write-Host "  status=$($script:ARCH.status) total_data=$($script:ARCH.total_data)"
  Step "request sinkronisasi dataset ke Hugging Face"
  $sync = Invoke-RestMethod -Method Post "$script:API/sync-dataset-to-hf" -ContentType "application/json" -Body ($script:ARCH | ConvertTo-Json -Depth 10)
  Write-Host "  uploaded=$($sync.total_uploaded) failed=$($sync.total_failed)"
  Step "susun payload training"
  # Sesuai train.py di HF Space (POST /train): version = nama model BARU, model_filename = model DASAR (model aktif) yang diunduh dari HF,
  # records[].image_path = path di HF Dataset (hasil sync), predicted_class dari arsip.
  $recs = @()
  foreach ($pth in $sync.uploaded) {
    $item = $script:ARCH.data | Where-Object { $_.processed_filename -eq (Split-Path $pth -Leaf) } | Select-Object -First 1
    if ($item) { $recs += @{ image_path=$pth; predicted_class=$item.predicted_class; source="new" } }
  }
  $script:TRAIN_BODY = @{ version=$script:NEW_VER; model_filename=$script:PREV_FILE; records=$recs } | ConvertTo-Json -Depth 6
  return $true
}

# ====== 3. BUILD MODEL ======
function build_model {
  Step "kirim perintah training ke HF Space (tunggu selesai)"
  $t = Invoke-RestMethod -Method Post "$script:HF_TRAIN$script:HF_TRAIN_PATH" -ContentType "application/json" -Body $script:TRAIN_BODY -TimeoutSec 7200
  $t | Out-Host
  Step "salin akurasi/loss/waktu dari hasil training ke payload register"
  # Field respons /train: accuracy ("xx.xx%"), final_loss, total_data_retrain, execution_time, model_filename
  $fn = if ($t.model_filename) { $t.model_filename } else { "$($script:NEW_VER).pth" }
  $body = @{ version=$script:NEW_VER; model_filename=$fn; accuracy=[string]$t.accuracy
             final_loss=[double]$t.final_loss; total_data_retrain=[int]$t.total_data_retrain; execution_time=[double]$t.execution_time } | ConvertTo-Json
  Step "kirim hasil ke backend (register model)"
  $script:REG = Invoke-RestMethod -Method Post "$script:API/register-trained-model" -ContentType "application/json" -Body $body
  $script:REG | Select-Object is_best_model,model_version,accuracy,baseline_accuracy,rejection_reason | Format-List | Out-Host
  return $true
}

# ====== 4. DEPLOYMENT ======
function deployment {
  Step "baca is_best_model & putuskan jalur"
  function Esc($s) { [uri]::EscapeDataString([string]$s) }
  if ($script:REG.is_best_model) {
    Write-Host "  -> DITERIMA" -ForegroundColor Green
    Step "hapus model lama"
    Invoke-RestMethod -Method Delete "$script:API/delete-old-model?filename=$(Esc $script:PREV_FILE)" | Out-Host
    Step "arsip citra ke Google Drive per kelas (MANUAL: buka Drive, upload, ulangi per kelas)"
    Pause-Manual "Upload ke Drive"
  } else {
    Write-Host "  -> DITOLAK" -ForegroundColor Red
    Step "hapus model kandidat"
    Invoke-RestMethod -Method Delete "$script:API/delete-rejected-model?filename=$(Esc $script:REG.model_path)" | Out-Host
    Step "tandai data sudah dipakai"
    Invoke-RestMethod -Method Post "$script:API/mark-used" | Select-Object status,marked | Out-Host
  }
  Step "hapus citra lokal (raw & processed) satu per satu"
  foreach ($d in $script:ARCH.data) {
    if ($d.raw_image)       { Invoke-RestMethod -Method Delete "$script:API/delete-raw-local?image_path=$(Esc $d.raw_image)" | Out-Null }
    if ($d.processed_image) { Invoke-RestMethod -Method Delete "$script:API/delete-processed-local?image_path=$(Esc $d.processed_image)" | Out-Null }
  }
  return $true
}

# ====== 5. MONITOR ======
function monitor {
  Step "catat hasil ke Google Sheets (versi/akurasi/baseline/status/alasan)"; Pause-Manual "Isi Sheets"
  Step "kirim notifikasi ke Telegram (ketik manual)";                      Pause-Manual "Kirim Telegram"
  return $true
}

function run_cycle {
  $c = [int]((& $script:DB_CMD) | Select-Object -Last 1)
  if ($c -gt ($script:MIN_DATA * 4)) { Write-Host "Data baru = $c (terlalu banyak, tidak sebanding dgn uji n8n). Jalankan dulu: prepare_data 5" -ForegroundColor Red; return }
  $script:STEPS = 0; $script:ROWS = @()
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $script:ERRS = 0
  $go = Run-Stage "1_Ingestion" { ingestion }
  if ($go) { $go = Run-Stage "2_Preparation" { preparation } }
  if ($go) { $go = Run-Stage "3_BuildModel"  { build_model } }
  if ($go) { $go = Run-Stage "4_Deployment"  { deployment } }
  if ($go) { $go = Run-Stage "5_Monitor"     { monitor } }
  $sw.Stop(); Show-Summary $sw.Elapsed.TotalSeconds
}
Write-Host "Siap. 1) prepare_data 5   2) run_cycle   (selesai semua uji: restore_data)" -ForegroundColor Green