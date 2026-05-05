# %% [markdown]
# # 🏋️ Biomechanical Squat Analysis — Classroom Dashboard
#
# **Purpose**: Process pose-detection CSV data and compute biomechanical
# exercise metrics grouped by student/track and repetition.
#
# Sections:
# 1. Setup & Imports
# 2. CSV Inspection
# 3. Schema Mapping
# 4. Data Cleaning
# 5. Feature Engineering
# 6. Repetition Segmentation
# 7. Biomechanical Flags
# 8. Aggregation
# 9. Visualizations
# 10. Outputs

# %% [markdown]
# ## 1 · Setup & Imports

# %%
# Cell 1 — Setup
import warnings, pathlib, glob, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import uniform_filter1d

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("ℹ️  plotly not installed — falling back to matplotlib only.")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.max_open_warning": 50})

print("✅ Libraries loaded.")

# %% [markdown]
# ### 1b · Load CSV
# Change `CSV_PATH` to point at your data.  When running in Colab you can
# upload a file or mount Google Drive.

# %%
# Cell 2 — File loading
# ── Option A: local path (default) ──────────────────────────────────────────
# Point to a single session CSV or a glob pattern for all sessions.
DATA_DIR = pathlib.Path("data/sessions")

# Auto-detect: use all frames.csv files found under DATA_DIR
csv_files = sorted(DATA_DIR.rglob("frames.csv"))
if not csv_files:
    # ── Option B: Colab upload ──────────────────────────────────────────────
    try:
        from google.colab import files as colab_files  # type: ignore
        print("📤 Upload your frames.csv file(s):")
        uploaded = colab_files.upload()
        csv_files = [pathlib.Path(k) for k in uploaded.keys()]
    except Exception:
        raise FileNotFoundError(
            "No CSV files found. Set DATA_DIR or upload via Colab."
        )

print(f"📂 Found {len(csv_files)} CSV file(s):")
for f in csv_files:
    print(f"   {f}  ({f.stat().st_size / 1e6:.1f} MB)")

# ── Load & concatenate ─────────────────────────────────────────────────────
dfs = []
for f in csv_files:
    tmp = pd.read_csv(f, low_memory=False)
    # Ensure session_id is populated (use parent folder name as fallback)
    if "session_id" in tmp.columns:
        mask = tmp["session_id"].isna() | (tmp["session_id"] == "")
        tmp.loc[mask, "session_id"] = f.parent.name
    else:
        tmp["session_id"] = f.parent.name
    dfs.append(tmp)

df_raw = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Combined dataframe: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

# %% [markdown]
# ## 2 · CSV Inspection

# %%
# Cell 3 — Inspect
def inspect_csv(df: pd.DataFrame) -> None:
    """Print shape, dtypes, nulls, and detected role columns."""
    print(f"Shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes.value_counts().to_string()}")
    print(f"\n{'Column':<35} {'Dtype':<12} {'Null%':>6}")
    print("-" * 55)
    for c in df.columns:
        pct = df[c].isna().mean() * 100
        print(f"{c:<35} {str(df[c].dtype):<12} {pct:5.1f}%")

    # Detect role columns
    id_candidates = [
        "student", "student_id", "id", "track_id",
        "person_id", "subject_id",
    ]
    rep_candidates = ["rep_num", "repetition", "rep_id", "rep_number"]
    time_candidates = ["frame_number", "frame", "timestamp", "time"]
    angle_cols = [c for c in df.columns if c.endswith("_deg")]
    vel_cols = [c for c in df.columns if "vy" in c or "speed" in c]
    kp_cols = [c for c in df.columns if c.startswith("kp") and "_norm" in c]

    print("\n── Detected role columns ──")
    for label, cands in [
        ("ID", id_candidates), ("Rep", rep_candidates), ("Time", time_candidates)
    ]:
        found = [c for c in cands if c in df.columns]
        print(f"  {label}: {found if found else '⚠️  NONE'}")
    print(f"  Angle columns ({len(angle_cols)}): {angle_cols[:5]}{'…' if len(angle_cols)>5 else ''}")
    print(f"  Velocity columns ({len(vel_cols)}): {vel_cols}")
    print(f"  Keypoint columns ({len(kp_cols)}): {kp_cols[:3]}{'…' if len(kp_cols)>3 else ''}")

inspect_csv(df_raw)

# %% [markdown]
# ## 3 · Schema Mapping Layer
#
# Maps real CSV column names to canonical names so the rest of the notebook
# is schema-independent.

# %%
# Cell 4 — Schema mapper
# ── Canonical name → list of possible source column names ───────────────────
_SCHEMA_VARIANTS: dict[str, list[str]] = {
    # Identity
    "student_id":     ["track_id", "person_id", "student_id", "student", "id", "subject_id"],
    "session_id":     ["session_id", "video_id", "session"],
    "frame_number":   ["frame_number", "frame", "frame_idx"],
    "timestamp":      ["timestamp", "time", "ts"],
    "fps":            ["fps"],
    # Labels
    "activity_label": ["activity_label", "activity"],
    "exercise_label": ["exercise_label", "exercise"],
    "phase_label":    ["phase_label", "phase"],
    "rep_id":         ["rep_id", "rep_num", "repetition", "rep_number"],
    # Angles
    "knee_left_deg":  ["knee_left_deg"],
    "knee_right_deg": ["knee_right_deg"],
    "hip_left_deg":   ["hip_left_deg"],
    "hip_right_deg":  ["hip_right_deg"],
    "trunk_deg":      ["trunk_deg"],
    "neck_deg":       ["neck_deg"],
    "pelvis_deg":     ["pelvis_deg"],
    "shoulder_left_deg":  ["shoulder_left_deg"],
    "shoulder_right_deg": ["shoulder_right_deg"],
    "ankle_left_deg":     ["ankle_left_deg"],
    "ankle_right_deg":    ["ankle_right_deg"],
    "elbow_left_deg":     ["elbow_left_deg"],
    "elbow_right_deg":    ["elbow_right_deg"],
    "shoulder_line_deg":  ["shoulder_line_deg"],
    "hip_line_deg":       ["hip_line_deg"],
    "left_arm_raise_deg": ["left_arm_raise_deg"],
    "right_arm_raise_deg":["right_arm_raise_deg"],
    "left_leg_abduction_deg":  ["left_leg_abduction_deg"],
    "right_leg_abduction_deg": ["right_leg_abduction_deg"],
    # Velocity
    "hip_center_y_norm":  ["hip_center_y_norm"],
    "hip_center_vy":      ["hip_center_vy"],
    "knee_left_vy":       ["knee_left_vy"],
    "knee_right_vy":      ["knee_right_vy"],
    "mean_joint_speed":   ["mean_joint_speed"],
    # Quality
    "mean_kpt_conf":      ["mean_kpt_conf"],
    "visible_kpt_count":  ["visible_kpt_count"],
    "is_valid_pose":      ["is_valid_pose"],
    "raw_confidence":     ["raw_confidence"],
    # Bbox
    "bbox_x1": ["bbox_x1"], "bbox_y1": ["bbox_y1"],
    "bbox_x2": ["bbox_x2"], "bbox_y2": ["bbox_y2"],
}

def build_schema_map(df: pd.DataFrame) -> dict[str, str | None]:
    """Return {canonical_name: actual_column_name_or_None}."""
    mapping: dict[str, str | None] = {}
    for canon, variants in _SCHEMA_VARIANTS.items():
        found = None
        for v in variants:
            if v in df.columns:
                found = v
                break
        mapping[canon] = found
    return mapping

def col(name: str) -> str | None:
    """Shortcut: return actual column name for a canonical name, or None."""
    return SCHEMA_MAP.get(name)

SCHEMA_MAP = build_schema_map(df_raw)

# Report
missing = [k for k, v in SCHEMA_MAP.items() if v is None]
print("✅ Schema mapping built.")
if missing:
    print(f"⚠️  Missing canonical columns: {missing}")
else:
    print("   All canonical columns found.")

# %% [markdown]
# ## 4 · Data Cleaning

# %%
# Cell 5 — Cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean frame-level pose data.

    Steps:
      1. Drop exact duplicate rows
      2. Sort by session → student → frame
      3. Filter invalid poses (if column exists)
      4. Remove low-confidence frames
      5. Convert boolean strings
      6. Log cleaning stats
    """
    n0 = len(df)
    df = df.copy()

    # 1. Duplicates
    df.drop_duplicates(inplace=True)
    n1 = len(df)

    # 2. Sort
    sort_cols = []
    for c in ["session_id", "student_id", "frame_number", "timestamp"]:
        actual = col(c)
        if actual and actual in df.columns:
            sort_cols.append(actual)
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 3. Boolean conversion (is_valid_pose may be string)
    vp = col("is_valid_pose")
    if vp and vp in df.columns:
        if df[vp].dtype == object:
            df[vp] = df[vp].map({"True": True, "False": False, "1": True, "0": False})
        df[vp] = df[vp].astype(bool)

    # 4. Filter invalid poses
    if vp and vp in df.columns:
        df = df[df[vp]].copy()
    n2 = len(df)

    # 5. Confidence filter — drop frames where mean keypoint conf < 0.3
    mc = col("mean_kpt_conf")
    if mc and mc in df.columns:
        df = df[df[mc] >= 0.3].copy()
    n3 = len(df)

    # 6. Filter to exercise frames only (skip idle/standing frames)
    al = col("activity_label")
    if al and al in df.columns:
        valid_labels = {"exercise", "squat", "front_jump"}
        mask = df[al].isin(valid_labels)
        if mask.sum() > 0:
            df = df[mask].copy()
    n4 = len(df)

    print(f"🧹 Cleaning report:")
    print(f"   Raw rows:            {n0:>8,}")
    print(f"   After dedup:         {n1:>8,}")
    print(f"   After valid-pose:    {n2:>8,}")
    print(f"   After conf ≥ 0.3:   {n3:>8,}")
    print(f"   After exercise-only: {n4:>8,}")
    return df

df = clean_data(df_raw)

# %% [markdown]
# ## 5 · Feature Engineering
#
# Compute derived biomechanical features from available columns.

# %%
# Cell 6 — Helper: safe column accessor
def gcol(df: pd.DataFrame, canon: str) -> pd.Series | None:
    """Get a column Series by canonical name, or None if missing."""
    actual = col(canon)
    if actual and actual in df.columns:
        return df[actual]
    return None


def safe_smooth(series: pd.Series, window: int = 7) -> pd.Series:
    """Savitzky-Golay smooth that handles NaN gracefully."""
    valid = series.dropna()
    if len(valid) < window:
        return series
    try:
        smoothed = savgol_filter(valid.values, min(window, len(valid) | 1), 2)
        out = series.copy()
        out.loc[valid.index] = smoothed
        return out
    except Exception:
        return series

# %%
# Cell 7 — Feature Engineering (vectorized)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived biomechanical columns to the frame-level dataframe.

    All operations are vectorized (no row-level loops).
    """
    df = df.copy()
    sid = col("student_id")
    sess = col("session_id")

    # ── A. Knee metrics ────────────────────────────────────────────────────
    kl = gcol(df, "knee_left_deg")
    kr = gcol(df, "knee_right_deg")
    if kl is not None and kr is not None:
        df["mean_knee_deg"] = df[[col("knee_left_deg"), col("knee_right_deg")]].mean(axis=1)
        df["knee_diff_deg"] = (kl - kr).abs()
    elif kl is not None:
        df["mean_knee_deg"] = kl
        df["knee_diff_deg"] = 0.0
    elif kr is not None:
        df["mean_knee_deg"] = kr
        df["knee_diff_deg"] = 0.0

    # ── B. Hip metrics ─────────────────────────────────────────────────────
    hl = gcol(df, "hip_left_deg")
    hr = gcol(df, "hip_right_deg")
    if hl is not None and hr is not None:
        df["mean_hip_deg"] = df[[col("hip_left_deg"), col("hip_right_deg")]].mean(axis=1)
        df["hip_diff_deg"] = (hl - hr).abs()
    elif hl is not None:
        df["mean_hip_deg"] = hl
    elif hr is not None:
        df["mean_hip_deg"] = hr

    # ── C. Trunk metrics ───────────────────────────────────────────────────
    trunk = gcol(df, "trunk_deg")
    if trunk is not None:
        # trunk_deg is angle from vertical (0 = upright).
        # A higher value = more forward lean.
        df["trunk_inclination"] = trunk
    else:
        # Fallback: compute from shoulder/hip keypoints if available
        sh_y_cols = [c for c in df.columns if c in ("kp5_y_norm", "kp6_y_norm")]
        hi_y_cols = [c for c in df.columns if c in ("kp11_y_norm", "kp12_y_norm")]
        sh_x_cols = [c for c in df.columns if c in ("kp5_x_norm", "kp6_x_norm")]
        hi_x_cols = [c for c in df.columns if c in ("kp11_x_norm", "kp12_x_norm")]
        if sh_y_cols and hi_y_cols and sh_x_cols and hi_x_cols:
            mid_sh_x = df[sh_x_cols].mean(axis=1)
            mid_sh_y = df[sh_y_cols].mean(axis=1)
            mid_hi_x = df[hi_x_cols].mean(axis=1)
            mid_hi_y = df[hi_y_cols].mean(axis=1)
            dx = mid_hi_x - mid_sh_x
            dy = mid_hi_y - mid_sh_y
            df["trunk_inclination"] = np.degrees(np.arctan2(dx.abs(), dy.abs() + 1e-6))

    # ── D. Smoothed signals per student (for velocity & segmentation) ──────
    hip_y = gcol(df, "hip_center_y_norm")
    if hip_y is not None and sid:
        df["hip_y_smooth"] = df.groupby([sess, sid] if sess else [sid])[
            col("hip_center_y_norm")
        ].transform(lambda s: safe_smooth(s, 9))

    if "mean_knee_deg" in df.columns and sid:
        df["knee_smooth"] = df.groupby([sess, sid] if sess else [sid])[
            "mean_knee_deg"
        ].transform(lambda s: safe_smooth(s, 9))

    print(f"✅ Feature engineering complete — {len(df.columns)} total columns.")
    new_cols = [c for c in df.columns if c not in df_raw.columns]
    print(f"   New columns: {new_cols}")
    return df

df = engineer_features(df)

# %% [markdown]
# ## 6 · Repetition Segmentation
#
# If `rep_id` already exists and is populated, validate it.
# Otherwise, infer reps from knee-angle valleys (bottom of squat).

# %%
# Cell 8 — Repetition segmentation
def infer_repetitions(
    group: pd.DataFrame,
    signal_col: str = "knee_smooth",
    fallback_col: str = "mean_knee_deg",
    min_depth_deg: float = 120.0,
    min_frames_between: int = 10,
) -> np.ndarray:
    """Detect squat repetitions from knee-angle signal for one student.

    A squat rep = the signal dips below *min_depth_deg* (valley).
    We find valleys using scipy.find_peaks on the *inverted* signal.

    Returns an array of rep labels (0 = no rep, 1..N = rep index).
    """
    n = len(group)
    reps = np.zeros(n, dtype=int)

    # Pick best available signal
    if signal_col in group.columns and group[signal_col].notna().sum() > 20:
        sig = group[signal_col].values.copy()
    elif fallback_col in group.columns and group[fallback_col].notna().sum() > 20:
        sig = group[fallback_col].values.copy()
    else:
        return reps  # can't segment

    # Fill NaN with interpolation for peak detection
    s = pd.Series(sig)
    s = s.interpolate(limit_direction="both").values

    # Invert signal: valleys in knee angle = peaks in inverted
    inverted = -s
    peaks, properties = find_peaks(
        inverted,
        height=-min_depth_deg,  # knee angle < min_depth_deg at valley
        distance=min_frames_between,
        prominence=5,  # at least 5° prominence
    )

    if len(peaks) == 0:
        return reps

    # Assign rep labels: each valley is the "bottom" of a rep.
    # A rep spans from the midpoint between consecutive valleys.
    boundaries = [0]
    for i in range(len(peaks) - 1):
        mid = (peaks[i] + peaks[i + 1]) // 2
        boundaries.append(mid)
    boundaries.append(n)

    for rep_idx in range(len(peaks)):
        start = boundaries[rep_idx]
        end = boundaries[rep_idx + 1]
        reps[start:end] = rep_idx + 1

    return reps


def segment_repetitions(df: pd.DataFrame) -> pd.DataFrame:
    """Add or validate rep_num column."""
    df = df.copy()
    sid = col("student_id")
    sess = col("session_id")
    rep_col = col("rep_id")

    group_cols = [c for c in [sess, sid] if c is not None]
    if not group_cols:
        print("⚠️  No student/session columns — skipping rep segmentation.")
        df["rep_num"] = 0
        return df

    # Check if existing rep_id is meaningful (has values > 0)
    has_existing_reps = False
    if rep_col and rep_col in df.columns:
        has_existing_reps = (df[rep_col] > 0).sum() > 10

    if has_existing_reps:
        print(f"✅ Using existing '{rep_col}' column ({df[rep_col].nunique()} unique values).")
        df["rep_num"] = df[rep_col]
    else:
        print("🔄 Inferring repetitions from knee-angle signal…")
        rep_arrays = []
        for _, grp in df.groupby(group_cols):
            reps = infer_repetitions(grp)
            rep_arrays.append(pd.Series(reps, index=grp.index))
        df["rep_num"] = pd.concat(rep_arrays).sort_index()
        n_reps = df.loc[df["rep_num"] > 0, "rep_num"].max()
        print(f"   Detected up to {n_reps} reps per student.")

    # Stats
    rep_stats = (
        df[df["rep_num"] > 0]
        .groupby(group_cols)["rep_num"]
        .nunique()
        .reset_index(name="n_reps")
    )
    print(f"\n📊 Reps per student:\n{rep_stats.to_string(index=False)}")
    return df

df = segment_repetitions(df)

# %% [markdown]
# ## 7 · Biomechanical Flags
#
# Rule-based detection for:
# - Insufficient depth
# - Strong L/R asymmetry
# - Excessive trunk lean
# - "Good morning" squat pattern
# - Fatigue deterioration

# %%
# Cell 9 — Flag helpers
def detect_good_morning(
    trunk_incl: float,
    knee_angle: float,
    trunk_standing: float = 10.0,
    trunk_threshold: float = 35.0,
    knee_min_for_squat: float = 110.0,
) -> bool:
    """Detect 'good morning' squat pattern.

    A good-morning occurs when:
      - Trunk inclination increases substantially (> trunk_threshold degrees)
      - Knee flexion is relatively limited (knee angle stays > knee_min_for_squat)
      - i.e. the person bends forward at the hips instead of squatting
    """
    if pd.isna(trunk_incl) or pd.isna(knee_angle):
        return False
    excessive_lean = trunk_incl > trunk_threshold
    limited_knee_flex = knee_angle > knee_min_for_squat
    return bool(excessive_lean and limited_knee_flex)


def compute_fatigue_index(student_reps: pd.DataFrame) -> pd.Series:
    """Compute a composite fatigue score across ordered reps for one student.

    Fatigue indicators (each normalized to z-score):
      1. Decreasing ascent velocity → higher fatigue
      2. Increasing trunk lean → higher fatigue
      3. Decreasing ROM → higher fatigue
      4. Increasing rep duration → higher fatigue
      5. Increasing asymmetry → higher fatigue

    Final score = mean of available z-scored indicators, then min-max scaled
    to [0, 1] where 1 = maximum fatigue.
    """
    if len(student_reps) < 3:
        return pd.Series(0.0, index=student_reps.index)

    indicators = pd.DataFrame(index=student_reps.index)

    # Higher ascent velocity = less fatigue → invert
    if "Ascent_Velocity" in student_reps.columns:
        v = student_reps["Ascent_Velocity"]
        if v.std() > 1e-6:
            indicators["vel"] = -(v - v.mean()) / (v.std() + 1e-9)

    # Higher trunk lean = more fatigue
    if "Peak_Trunk_Inclination" in student_reps.columns:
        t = student_reps["Peak_Trunk_Inclination"]
        if t.std() > 1e-6:
            indicators["trunk"] = (t - t.mean()) / (t.std() + 1e-9)

    # Lower ROM = more fatigue → invert
    if "Real_ROM" in student_reps.columns:
        r = student_reps["Real_ROM"]
        if r.std() > 1e-6:
            indicators["rom"] = -(r - r.mean()) / (r.std() + 1e-9)

    # Longer rep = more fatigue
    if "Rep_Duration" in student_reps.columns:
        d = student_reps["Rep_Duration"]
        if d.std() > 1e-6:
            indicators["dur"] = (d - d.mean()) / (d.std() + 1e-9)

    # Higher asymmetry = more fatigue
    if "Leg_Asymmetry" in student_reps.columns:
        a = student_reps["Leg_Asymmetry"]
        if a.std() > 1e-6:
            indicators["asym"] = (a - a.mean()) / (a.std() + 1e-9)

    if indicators.empty or len(indicators.columns) == 0:
        return pd.Series(0.0, index=student_reps.index)

    raw = indicators.mean(axis=1)
    # Min-max scale to [0, 1]
    rng = raw.max() - raw.min()
    if rng > 1e-9:
        return (raw - raw.min()) / rng
    return pd.Series(0.0, index=student_reps.index)

print("✅ Flag functions defined.")

# %% [markdown]
# ## 8 · Aggregation
#
# One row per student × repetition, plus student-level summaries.

# %%
# Cell 10 — Per-rep aggregation
def compute_rep_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frame-level data into one row per student × rep.

    Metrics computed:
      - Max/Min/Mean knee angle
      - Real ROM (max − min knee angle)
      - Ascent/Descent velocity (from hip_center_vy)
      - Rep duration (seconds)
      - Bottom pause (frames near minimum knee angle)
      - Leg asymmetry (mean |left − right| knee angle)
      - Peak trunk inclination
      - Good-morning flag
    """
    sid = col("student_id")
    sess = col("session_id")
    group_cols = [c for c in [sess, sid] if c is not None]

    # Only aggregate reps > 0
    rep_df = df[df["rep_num"] > 0].copy()
    if rep_df.empty:
        print("⚠️  No reps found — cannot aggregate.")
        return pd.DataFrame()

    agg_rows = []
    for keys, grp in rep_df.groupby(group_cols + ["rep_num"]):
        row = {}
        # Unpack keys
        if len(group_cols) == 2:
            row["Session"] = keys[0]
            row["Student"] = keys[1]
            row["Rep_Num"] = keys[2]
        elif len(group_cols) == 1:
            row["Student"] = keys[0]
            row["Rep_Num"] = keys[1]
        else:
            row["Rep_Num"] = keys[0]

        n_frames = len(grp)
        row["N_Frames"] = n_frames

        # ── Knee metrics ───────────────────────────────────────────────────
        knee = grp["mean_knee_deg"] if "mean_knee_deg" in grp.columns else None
        if knee is not None and knee.notna().sum() > 0:
            row["Max_Knee_Angle"] = round(knee.max(), 1)
            row["Min_Knee_Angle"] = round(knee.min(), 1)
            row["Mean_Knee_Angle"] = round(knee.mean(), 1)
            row["Real_ROM"] = round(knee.max() - knee.min(), 1)
        else:
            row["Max_Knee_Angle"] = np.nan
            row["Min_Knee_Angle"] = np.nan
            row["Mean_Knee_Angle"] = np.nan
            row["Real_ROM"] = np.nan

        # ── Velocity metrics ───────────────────────────────────────────────
        hip_vy_col = col("hip_center_vy")
        if hip_vy_col and hip_vy_col in grp.columns:
            vy = grp[hip_vy_col].dropna()
            if len(vy) > 2 and knee is not None and knee.notna().sum() > 0:
                bottom_idx = knee.idxmin()
                # Descent = before bottom (vy typically positive in image coords = downward)
                descent = vy.loc[:bottom_idx]
                ascent = vy.loc[bottom_idx:]
                row["Descent_Velocity"] = round(descent.mean(), 5) if len(descent) > 0 else np.nan
                row["Ascent_Velocity"] = round(ascent.mean(), 5) if len(ascent) > 0 else np.nan
            else:
                row["Descent_Velocity"] = np.nan
                row["Ascent_Velocity"] = np.nan
        else:
            row["Descent_Velocity"] = np.nan
            row["Ascent_Velocity"] = np.nan

        # ── Duration ───────────────────────────────────────────────────────
        ts_col = col("timestamp")
        fps_col = col("fps")
        if ts_col and ts_col in grp.columns:
            ts = grp[ts_col]
            row["Rep_Duration"] = round(ts.max() - ts.min(), 3)
        elif fps_col and fps_col in grp.columns:
            fps_val = grp[fps_col].median()
            row["Rep_Duration"] = round(n_frames / max(fps_val, 1), 3) if fps_val > 0 else np.nan
        else:
            row["Rep_Duration"] = np.nan

        # ── Bottom pause (frames within 5° of min knee angle) ─────────────
        if knee is not None and knee.notna().sum() > 0:
            min_k = knee.min()
            near_bottom = (knee <= min_k + 5).sum()
            row["Bottom_Pause"] = int(near_bottom)
        else:
            row["Bottom_Pause"] = 0

        # ── Asymmetry ─────────────────────────────────────────────────────
        if "knee_diff_deg" in grp.columns:
            row["Leg_Asymmetry"] = round(grp["knee_diff_deg"].mean(), 2)
        else:
            row["Leg_Asymmetry"] = np.nan

        # ── Trunk ─────────────────────────────────────────────────────────
        if "trunk_inclination" in grp.columns:
            trunk_vals = grp["trunk_inclination"].dropna()
            row["Peak_Trunk_Inclination"] = round(trunk_vals.max(), 1) if len(trunk_vals) > 0 else np.nan
            row["Mean_Trunk_Inclination"] = round(trunk_vals.mean(), 1) if len(trunk_vals) > 0 else np.nan
        else:
            row["Peak_Trunk_Inclination"] = np.nan
            row["Mean_Trunk_Inclination"] = np.nan

        # ── Good Morning flag ─────────────────────────────────────────────
        if "trunk_inclination" in grp.columns and knee is not None:
            peak_trunk = row.get("Peak_Trunk_Inclination", 0) or 0
            min_knee = row.get("Min_Knee_Angle", 180) or 180
            row["GoodMorning_Flag"] = detect_good_morning(peak_trunk, min_knee)
        else:
            row["GoodMorning_Flag"] = False

        # ── Quality flags ─────────────────────────────────────────────────
        flags = []
        if row.get("Real_ROM") is not None and row["Real_ROM"] < 40:
            flags.append("low_ROM")
        if row.get("Leg_Asymmetry") is not None and row["Leg_Asymmetry"] > 15:
            flags.append("high_asymmetry")
        if row.get("Peak_Trunk_Inclination") is not None and row["Peak_Trunk_Inclination"] > 40:
            flags.append("excessive_lean")
        if row.get("GoodMorning_Flag"):
            flags.append("good_morning")
        if row.get("Min_Knee_Angle") is not None and row["Min_Knee_Angle"] > 120:
            flags.append("insufficient_depth")
        row["Quality_Flags"] = "; ".join(flags) if flags else "ok"

        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    # ── Fatigue index (per student) ────────────────────────────────────────
    student_col = "Student"
    session_col = "Session" if "Session" in agg_df.columns else None
    fatigue_group = [c for c in [session_col, student_col] if c and c in agg_df.columns]

    if fatigue_group:
        fatigue_scores = []
        for _, sgrp in agg_df.groupby(fatigue_group):
            sgrp_sorted = sgrp.sort_values("Rep_Num")
            fi = compute_fatigue_index(sgrp_sorted)
            fatigue_scores.append(fi)
        agg_df["Fatigue_Index"] = pd.concat(fatigue_scores).reindex(agg_df.index).round(3)
    else:
        agg_df["Fatigue_Index"] = 0.0

    print(f"✅ Aggregated {len(agg_df)} rep rows.")
    return agg_df

agg = compute_rep_metrics(df)
if not agg.empty:
    display(agg.head(15)) if hasattr(__builtins__, '__IPYTHON__') else print(agg.head(15).to_string())

# %%
# Cell 11 — Student-level summary
def compute_student_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """Compute student-level summary from per-rep aggregation."""
    if agg.empty:
        return pd.DataFrame()

    student_col = "Student"
    session_col = "Session" if "Session" in agg.columns else None
    grp_cols = [c for c in [session_col, student_col] if c and c in agg.columns]

    if not grp_cols:
        return pd.DataFrame()

    summary_rows = []
    for keys, sdf in agg.groupby(grp_cols):
        row = {}
        if isinstance(keys, tuple):
            for c, k in zip(grp_cols, keys):
                row[c] = k
        else:
            row[grp_cols[0]] = keys

        sdf_sorted = sdf.sort_values("Rep_Num")

        row["Total_Reps"] = len(sdf)
        row["Mean_ROM"] = round(sdf["Real_ROM"].mean(), 1) if "Real_ROM" in sdf.columns else np.nan
        row["Std_ROM"] = round(sdf["Real_ROM"].std(), 1) if "Real_ROM" in sdf.columns else np.nan
        row["Mean_Ascent_Vel"] = round(sdf["Ascent_Velocity"].mean(), 5) if "Ascent_Velocity" in sdf.columns else np.nan
        row["Mean_Asymmetry"] = round(sdf["Leg_Asymmetry"].mean(), 2) if "Leg_Asymmetry" in sdf.columns else np.nan
        row["Mean_Trunk_Incl"] = round(sdf["Peak_Trunk_Inclination"].mean(), 1) if "Peak_Trunk_Inclination" in sdf.columns else np.nan

        # Trends (slope of linear fit across reps)
        if len(sdf_sorted) >= 3:
            x = np.arange(len(sdf_sorted))
            for metric, label in [
                ("Real_ROM", "ROM_Trend"),
                ("Ascent_Velocity", "Vel_Trend"),
                ("Leg_Asymmetry", "Asym_Trend"),
                ("Peak_Trunk_Inclination", "Trunk_Trend"),
            ]:
                if metric in sdf_sorted.columns:
                    vals = sdf_sorted[metric].dropna()
                    if len(vals) >= 3:
                        slope = np.polyfit(np.arange(len(vals)), vals.values, 1)[0]
                        row[label] = round(slope, 4)
                    else:
                        row[label] = np.nan
                else:
                    row[label] = np.nan
        else:
            row["ROM_Trend"] = np.nan
            row["Vel_Trend"] = np.nan
            row["Asym_Trend"] = np.nan
            row["Trunk_Trend"] = np.nan

        row["Fatigue_Final"] = round(sdf_sorted["Fatigue_Index"].iloc[-1], 3) if "Fatigue_Index" in sdf.columns and len(sdf_sorted) > 0 else np.nan
        row["Flagged_Reps"] = (sdf["Quality_Flags"] != "ok").sum() if "Quality_Flags" in sdf.columns else 0

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print(f"✅ Student summary: {len(summary)} students.")
    return summary

student_summary = compute_student_summary(agg)
if not student_summary.empty:
    display(student_summary) if hasattr(__builtins__, '__IPYTHON__') else print(student_summary.to_string())

# %% [markdown]
# ## 9 · Visualizations

# %%
# Cell 12 — Class-level plots
def plot_class_overview(agg: pd.DataFrame) -> None:
    """Generate class-wide distribution and trend plots."""
    if agg.empty:
        print("⚠️  No data to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Class Overview — Squat Biomechanics", fontsize=16, fontweight="bold")

    # 1. ROM distribution
    ax = axes[0, 0]
    if "Real_ROM" in agg.columns:
        agg["Real_ROM"].dropna().hist(bins=20, ax=ax, color="#4C72B0", edgecolor="white")
        ax.axvline(agg["Real_ROM"].median(), color="red", ls="--", label=f'Median={agg["Real_ROM"].median():.0f}°')
        ax.legend()
    ax.set_title("ROM Distribution")
    ax.set_xlabel("ROM (degrees)")

    # 2. Ascent velocity distribution
    ax = axes[0, 1]
    if "Ascent_Velocity" in agg.columns:
        vals = agg["Ascent_Velocity"].dropna()
        vals.hist(bins=20, ax=ax, color="#55A868", edgecolor="white")
        ax.axvline(vals.median(), color="red", ls="--", label=f"Median={vals.median():.4f}")
        ax.legend()
    ax.set_title("Ascent Velocity Distribution")
    ax.set_xlabel("Hip Vy (norm/frame)")

    # 3. Asymmetry distribution
    ax = axes[0, 2]
    if "Leg_Asymmetry" in agg.columns:
        agg["Leg_Asymmetry"].dropna().hist(bins=20, ax=ax, color="#C44E52", edgecolor="white")
        ax.axvline(15, color="orange", ls="--", label="Threshold=15°")
        ax.legend()
    ax.set_title("Leg Asymmetry Distribution")
    ax.set_xlabel("|L−R| Knee (degrees)")

    # 4. Fatigue trend across reps
    ax = axes[1, 0]
    if "Fatigue_Index" in agg.columns and "Rep_Num" in agg.columns:
        mean_fatigue = agg.groupby("Rep_Num")["Fatigue_Index"].mean()
        ax.plot(mean_fatigue.index, mean_fatigue.values, "o-", color="#8172B2", linewidth=2)
        ax.fill_between(mean_fatigue.index, mean_fatigue.values, alpha=0.2, color="#8172B2")
    ax.set_title("Mean Fatigue Across Reps")
    ax.set_xlabel("Rep #")
    ax.set_ylabel("Fatigue Index")

    # 5. ROM vs Ascent Velocity scatter
    ax = axes[1, 1]
    if "Real_ROM" in agg.columns and "Ascent_Velocity" in agg.columns:
        ax.scatter(agg["Real_ROM"], agg["Ascent_Velocity"], alpha=0.5, c="#DD8452", edgecolors="k", s=40)
    ax.set_title("ROM vs Ascent Velocity")
    ax.set_xlabel("ROM (degrees)")
    ax.set_ylabel("Ascent Velocity")

    # 6. Trunk Inclination vs Asymmetry
    ax = axes[1, 2]
    if "Peak_Trunk_Inclination" in agg.columns and "Leg_Asymmetry" in agg.columns:
        colors = ["red" if f != "ok" else "#4C72B0" for f in agg.get("Quality_Flags", ["ok"] * len(agg))]
        ax.scatter(agg["Peak_Trunk_Inclination"], agg["Leg_Asymmetry"], alpha=0.5, c=colors, edgecolors="k", s=40)
    ax.set_title("Trunk Incl. vs Asymmetry")
    ax.set_xlabel("Peak Trunk (degrees)")
    ax.set_ylabel("Leg Asymmetry (degrees)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("class_overview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Class overview saved → class_overview.png")


plot_class_overview(agg)

# %%
# Cell 13 — Per-student plots
def plot_student_dashboard(agg: pd.DataFrame, max_students: int = 10) -> None:
    """Per-student trend plots across repetitions."""
    if agg.empty:
        return

    student_col = "Student"
    if student_col not in agg.columns:
        print("⚠️  No Student column — skipping per-student plots.")
        return

    students = agg[student_col].unique()[:max_students]
    metrics = [
        ("Real_ROM", "ROM (°)", "#4C72B0"),
        ("Ascent_Velocity", "Ascent Vel.", "#55A868"),
        ("Fatigue_Index", "Fatigue", "#C44E52"),
        ("Peak_Trunk_Inclination", "Trunk Incl. (°)", "#8172B2"),
        ("Leg_Asymmetry", "Asymmetry (°)", "#DD8452"),
    ]
    available = [(m, l, c) for m, l, c in metrics if m in agg.columns]

    if not available:
        print("⚠️  No metrics available for per-student plots.")
        return

    n_metrics = len(available)
    for student in students:
        sdf = agg[agg[student_col] == student].sort_values("Rep_Num")
        if len(sdf) < 2:
            continue

        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 3.5))
        if n_metrics == 1:
            axes = [axes]
        fig.suptitle(f"Student {student}", fontsize=13, fontweight="bold")

        for ax, (metric, label, color) in zip(axes, available):
            vals = sdf[metric].values
            reps = sdf["Rep_Num"].values
            ax.plot(reps, vals, "o-", color=color, linewidth=2, markersize=5)
            ax.fill_between(reps, vals, alpha=0.15, color=color)
            ax.set_xlabel("Rep #")
            ax.set_ylabel(label)
            ax.set_title(label)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(f"student_{student}_dashboard.png", dpi=120, bbox_inches="tight")
        plt.show()

    print(f"📊 Generated dashboards for {len(students)} students.")


plot_student_dashboard(agg)

# %%
# Cell 14 — Technique-focused plots
def plot_technique_analysis(agg: pd.DataFrame, df_frame: pd.DataFrame) -> None:
    """Technique-focused visualizations: good-morning reps, warnings bar chart."""
    if agg.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Technique Analysis", fontsize=15, fontweight="bold")

    # 1. Good-morning flagged reps
    ax = axes[0]
    if "GoodMorning_Flag" in agg.columns:
        gm_counts = agg.groupby("Student")["GoodMorning_Flag"].sum().sort_values(ascending=False).head(15)
        if gm_counts.sum() > 0:
            gm_counts.plot.barh(ax=ax, color="#C44E52", edgecolor="white")
            ax.set_xlabel("# Good-Morning Reps")
        else:
            ax.text(0.5, 0.5, "No good-morning\npatterns detected", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Good-Morning Detections")

    # 2. Bar chart: students with most warnings
    ax = axes[1]
    if "Quality_Flags" in agg.columns and "Student" in agg.columns:
        flagged = agg[agg["Quality_Flags"] != "ok"].groupby("Student").size().sort_values(ascending=False).head(15)
        if len(flagged) > 0:
            flagged.plot.barh(ax=ax, color="#DD8452", edgecolor="white")
            ax.set_xlabel("# Flagged Reps")
        else:
            ax.text(0.5, 0.5, "All reps passed\nquality checks", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Students with Most Warnings")

    # 3. Trunk inclination profile (mean across all students per rep)
    ax = axes[2]
    if "Peak_Trunk_Inclination" in agg.columns and "Rep_Num" in agg.columns:
        trunk_by_rep = agg.groupby("Rep_Num")["Peak_Trunk_Inclination"].agg(["mean", "std"])
        ax.plot(trunk_by_rep.index, trunk_by_rep["mean"], "o-", color="#8172B2", linewidth=2)
        ax.fill_between(trunk_by_rep.index,
                        trunk_by_rep["mean"] - trunk_by_rep["std"],
                        trunk_by_rep["mean"] + trunk_by_rep["std"],
                        alpha=0.2, color="#8172B2")
        ax.axhline(35, color="red", ls="--", alpha=0.7, label="Lean threshold")
        ax.legend()
    ax.set_title("Trunk Inclination Trend")
    ax.set_xlabel("Rep #")
    ax.set_ylabel("Peak Trunk (°)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("technique_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Technique analysis saved → technique_analysis.png")


plot_technique_analysis(agg, df)

# %% [markdown]
# ## 10 · Outputs

# %%
# Cell 15 — Save CSVs
OUTPUT_DIR = pathlib.Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Cleaned frame-level
out_frames = OUTPUT_DIR / "cleaned_frame_level.csv"
df.to_csv(out_frames, index=False)
print(f"💾 {out_frames}  ({len(df):,} rows)")

# 2. Aggregated reps
if not agg.empty:
    out_agg = OUTPUT_DIR / "aggregated_reps.csv"
    agg.to_csv(out_agg, index=False)
    print(f"💾 {out_agg}  ({len(agg):,} rows)")

# 3. Student summary
if not student_summary.empty:
    out_sum = OUTPUT_DIR / "student_summary.csv"
    student_summary.to_csv(out_sum, index=False)
    print(f"💾 {out_sum}  ({len(student_summary):,} rows)")

# 4. Flagged reps only
if not agg.empty and "Quality_Flags" in agg.columns:
    flagged = agg[agg["Quality_Flags"] != "ok"]
    out_flag = OUTPUT_DIR / "flagged_reps.csv"
    flagged.to_csv(out_flag, index=False)
    print(f"💾 {out_flag}  ({len(flagged):,} flagged rows)")

print("\n✅ All outputs saved to ./output/")

# %% [markdown]
# ## Summary of Metrics
#
# | Metric | Source | Status |
# |---|---|---|
# | `Max/Min/Mean_Knee_Angle` | `knee_left_deg`, `knee_right_deg` | ✅ Direct |
# | `Real_ROM` | max − min knee angle per rep | ✅ Computed |
# | `Ascent/Descent_Velocity` | `hip_center_vy` split at bottom | ✅ Computed |
# | `Rep_Duration` | `timestamp` range per rep | ✅ Computed |
# | `Bottom_Pause` | frames within 5° of min knee | ✅ Computed |
# | `Leg_Asymmetry` | `|knee_left − knee_right|` | ✅ Computed |
# | `Peak_Trunk_Inclination` | `trunk_deg` | ✅ Direct |
# | `GoodMorning_Flag` | trunk + knee rule | ✅ Computed |
# | `Fatigue_Index` | composite z-score trend | ✅ Computed |
# | `Quality_Flags` | multi-rule flags | ✅ Computed |
# | `rep_num` | `rep_id` from CSV (or inferred) | ✅ Direct/Inferred |
#
# ### Adapting to different CSV schemas
# Edit the `_SCHEMA_VARIANTS` dictionary in Cell 4 to add your column name
# variants.  The rest of the notebook uses `col("canonical_name")` and will
# adapt automatically.

