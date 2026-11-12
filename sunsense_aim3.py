"""
SunSense - Aim 3: dashboard, reminders, and replay/scenario testing
Owner: Rajiv Dahal

What this does:
  1. Uses the Aim 2 pipeline (sunsense_aim2.py) to build features and get a
     next-5-min forecast category for each interval (Random Forest trained on
     the training block only; persistence fallback when history is too short).
  2. Runs the reminder engine on a replay of the dataset:
       - reminder when CURRENT UVI >= 3 or PREDICTED category >= Moderate
       - "sustained" reminder after >= 30 min of consecutive UVI >= 3
       - 10-minute cooldown between reminders
       - missing / invalid data -> no reminder, status "data unavailable"
     (Timing values are engineering settings, NOT safe-exposure limits.)
  3. Runs 6 scripted scenario tests (low, moderate, high, prolonged,
     transition, missing-data) with expected outcomes and pass/fail.
  4. Measures software processing latency per replay step.
  5. Writes a self-contained HTML replay dashboard + a static PNG figure.

IMPORTANT: the input dataset is SYNTHETIC. Replayed sequences test the
software pipeline only; they are not sensor measurements or ML evidence.

Usage (terminal):  python sunsense_aim3.py --data sunsense_synthetic_uv_dataset.csv --out results
Usage (Jupyter):   from sunsense_aim3 import main; main("sunsense_synthetic_uv_dataset.csv", "results")
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sunsense_aim2 as a2

THRESH_UVI = 3.0          # reminder when current or predicted UVI category >= Moderate
SUSTAINED_MIN = 30        # sustained moderate-or-higher exposure reminder
COOLDOWN_MIN = 10         # minimum minutes between reminders
CAT_COLORS = ["#3a9d23", "#e8c51a", "#f08a24", "#d62728", "#7b3fa0"]


# --------------------------------------------------------------------------
# Reminder engine (pure function of the stream -> easy to test and replay)
# --------------------------------------------------------------------------
def reminder_engine(stream):
    """
    stream: DataFrame with columns timestamp, uvi (NaN = missing/invalid),
            pred_cat (int category index or -1 if unavailable)
    returns: DataFrame with status, sustained minutes, reminder flag/type/reason
    """
    last_fire = None
    run_min = 0
    rows = []
    for _, r in stream.iterrows():
        t, u, pc = r["timestamp"], r["uvi"], int(r["pred_cat"])
        valid = np.isfinite(u) and u >= 0
        rec = {"timestamp": t, "uvi": u if valid else np.nan, "pred_cat": pc,
               "data_status": "ok" if valid else "data unavailable",
               "reminder": False, "reminder_type": "", "reason": "",
               "suppressed_by_cooldown": False}
        if not valid:
            run_min = 0                      # never bridge a gap
            rec["sustained_min"] = 0
            rec["reason"] = "suppressed: missing/invalid data"
            rows.append(rec)
            continue

        run_min = run_min + a2.STEP_MIN if u >= THRESH_UVI else 0
        rec["sustained_min"] = run_min
        reasons = []
        if run_min >= SUSTAINED_MIN:
            reasons.append("sustained")
        if u >= THRESH_UVI:
            reasons.append("current")
        if pc >= 1:                          # forecast Moderate or higher
            reasons.append("predicted")

        if reasons:
            since = None if last_fire is None else (t - last_fire).total_seconds() / 60
            if since is None or since >= COOLDOWN_MIN:
                rec["reminder"] = True
                rec["reminder_type"] = reasons[0]   # priority: sustained > current > predicted
                rec["reason"] = "+".join(reasons)
                last_fire = t
            else:
                rec["suppressed_by_cooldown"] = True
                rec["reason"] = "cooldown (" + "+".join(reasons) + ")"
        rows.append(rec)
    return pd.DataFrame(rows)


MESSAGES = {
    "current": "UV is Moderate or higher right now. Consider shade, a hat, or sunscreen.",
    "predicted": "UV is forecast to reach Moderate or higher in the next 5 minutes.",
    "sustained": "You have had 30+ minutes of Moderate-or-higher UV. Consider a shade break.",
}


# --------------------------------------------------------------------------
# Forecasts from Aim 2 (model trained on the training block only)
# --------------------------------------------------------------------------
def add_forecasts(df):
    feat = a2.build_features(df)
    feat, _ = a2.chronological_split(feat)
    rows = feat[feat["eligible"] & (feat["y"] >= 0)].dropna(subset=a2.FULL)
    tr = rows[rows.split == "train"]
    va = rows[rows.split == "val"]
    labels = sorted(rows["y"].unique())
    params, _ = a2.tune("rf", a2.FULL, tr, va, labels)

    has_hist = feat[a2.FULL].notna().all(axis=1)
    feat["pred_cat"] = feat["current_cat"]          # persistence fallback
    feat["pred_source"] = "persistence (short history)"
    X = feat.loc[has_hist, a2.FULL].values
    feat.loc[has_hist, "pred_cat"] = a2.fit_predict("rf", params, tr[a2.FULL].values,
                                                    tr["y"].values, X)
    feat.loc[has_hist, "pred_source"] = np.where(
        feat.loc[has_hist, "split"] == "train", "RF (in-sample)", "RF (held-out)")
    return feat, params


# --------------------------------------------------------------------------
# Scenario tests (scripted fixtures with expected outcomes)
# --------------------------------------------------------------------------
def _fixture(uvis, preds=None):
    t = pd.date_range("2026-11-10 12:00", periods=len(uvis), freq="5min")
    if preds is None:
        preds = [int(a2.uvi_to_cat([u])[0]) if np.isfinite(u) else -1 for u in uvis]
    return pd.DataFrame({"timestamp": t, "uvi": uvis, "pred_cat": preds})


def fire_minutes(out):
    t0 = out["timestamp"].iloc[0]
    return [int((t - t0).total_seconds() / 60) for t in out.loc[out.reminder, "timestamp"]]


def run_scenarios():
    tests = []

    def add(name, setup, expected, ok, out):
        tests.append({"scenario": name, "input": setup, "expected": expected,
                      "observed_fire_minutes": fire_minutes(out),
                      "observed_types": out.loc[out.reminder, "reminder_type"].tolist(),
                      "pass": bool(ok)})

    o = reminder_engine(_fixture([1.0] * 12))
    add("Low", "UVI 1.0 for 60 min", "no reminders", o.reminder.sum() == 0, o)

    o = reminder_engine(_fixture([4.0] * 4))
    add("Moderate", "UVI 4.0 for 20 min", "fire at 0 and 10 min (cooldown suppresses 5, 15)",
        fire_minutes(o) == [0, 10] and o.suppressed_by_cooldown.sum() == 2, o)

    o = reminder_engine(_fixture([7.0] * 4))
    add("High", "UVI 7.0 for 20 min", "fire at 0 and 10 min",
        fire_minutes(o) == [0, 10], o)

    o = reminder_engine(_fixture([4.0] * 9))
    st = o.loc[o.reminder_type == "sustained", "timestamp"]
    first_sustained = int((st.iloc[0] - o.timestamp.iloc[0]).total_seconds() / 60) if len(st) else None
    add("Prolonged", "UVI 4.0 for 45 min",
        "'sustained' type only once 30 min of exposure is reached (first at 30 min)",
        first_sustained == 30 and (o.loc[o.sustained_min < 30, "reminder_type"] != "sustained").all(), o)

    o = reminder_engine(_fixture([1.0, 1.0, 1.5, 4.5, 4.5, 1.0, 1.0, 1.0],
                                 preds=[0, 0, 1, 1, 0, 0, 0, 0]))
    add("Transition", "Low, forecast rises at 10 min, UVI 4.5 at 15-20 min, then shade",
        "early 'predicted' reminder at 10 min; cooldown at 15; fire at 20; none in shade",
        fire_minutes(o) == [10, 20] and o.reminder_type.iloc[2] == "predicted", o)

    o = reminder_engine(_fixture([4.0, 4.0, np.nan, np.nan, 4.0, 4.0, 4.0, 4.0]))
    add("Missing data", "UVI 4.0 with a 10-min gap (2 missing intervals)",
        "no reminder during gap; status 'data unavailable'; sustained counter resets",
        (~o.loc[2:3, "reminder"]).all()
        and (o.loc[2:3, "data_status"] == "data unavailable").all()
        and o.loc[4, "sustained_min"] == 5, o)
    return pd.DataFrame(tests)


# --------------------------------------------------------------------------
# Dashboard outputs
# --------------------------------------------------------------------------
def static_figure(tl, path):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(tl["timestamp"], tl["uvi"], "-", color="#555", lw=1.2, label="UVI proxy (current)")
    ax.scatter(tl["timestamp"], tl["uvi"], c=[CAT_COLORS[max(c, 0)] for c in tl["pred_cat"]],
               s=14, zorder=3, label="dot color = forecast category")
    fired = tl[tl.reminder]
    for typ, mk in [("current", "v"), ("predicted", "^"), ("sustained", "s")]:
        d = fired[fired.reminder_type == typ]
        if len(d):
            ax.scatter(d["timestamp"], [11.6] * len(d), marker=mk, s=28, color="#111",
                       label=f"reminder: {typ} ({len(d)})")
    ax.axhline(THRESH_UVI, ls="--", lw=0.9, color="#e8a317")
    ax.set_ylim(-0.3, 12.4)
    ax.set_ylabel("UVI proxy")
    ax.set_title("SYNTHETIC replay - reminder engine output (pipeline test only)", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SunSense dashboard - synthetic replay</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mut:#6b6a66;--line:#e4e2dc}
*{box-sizing:border-box}body{margin:0;font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:14px 20px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:18px;margin:0}.badge{background:#fff3cd;border:1px solid #e8c51a;border-radius:6px;padding:3px 8px;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.lab{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.big{font-size:32px;font-weight:650;margin-top:4px}.cat{display:inline-block;padding:2px 8px;border-radius:12px;color:#fff;font-size:13px;margin-top:4px}
#alert{margin:12px 20px;padding:12px 14px;border-radius:10px;border:1px solid var(--line);background:#fff;min-height:48px}
#alert.on{background:#fde8e8;border-color:#d62728}
.row{display:grid;grid-template-columns:2fr 1fr;gap:12px;padding:0 20px 20px}
canvas{width:100%;height:260px}#log{height:260px;overflow:auto;font-size:13px}
#log div{border-bottom:1px solid var(--line);padding:4px 0}
.ctl{display:flex;gap:8px;align-items:center}button{font:inherit;padding:5px 12px;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer}
input[type=range]{width:220px}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.row{grid-template-columns:1fr}}
</style></head><body>
<header><h1>SunSense dashboard</h1>
<span class="badge">SYNTHETIC replay - not sensor measurements, not a safe-exposure limit</span>
<div class="ctl"><button id="play">Play</button><input id="pos" type="range" min="0" value="0">
<span id="clock" class="lab"></span></div></header>
<div class="grid">
 <div class="card"><div class="lab">Current UVI (5-min)</div><div class="big" id="uvi">-</div><span class="cat" id="ccat"></span></div>
 <div class="card"><div class="lab">Forecast, next 5 min</div><div class="big" id="fc">-</div><div class="lab" id="fsrc"></div></div>
 <div class="card"><div class="lab">Cumulative dose at sensor</div><div class="big" id="dose">-</div><div class="lab">J/m&sup2; (estimate)</div></div>
 <div class="card"><div class="lab">Data quality</div><div class="big" id="dq" style="font-size:22px">-</div><div class="lab" id="sus"></div></div>
</div>
<div id="alert">No reminder.</div>
<div class="row"><div class="card"><div class="lab">UVI timeline (dots = forecast category, bars = reminders)</div><canvas id="cv"></canvas></div>
<div class="card"><div class="lab">Reminder log</div><div id="log"></div></div></div>
<script>
const D=__DATA__;const CATS=["Low","Moderate","High","Very high","Extreme"],COL=__COLORS__;
const $=id=>document.getElementById(id);let i=0,timer=null;$("pos").max=D.length-1;
const MSG=__MSG__;
function draw(){const c=$("cv"),r=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=r.width*dpr;c.height=r.height*dpr;
const g=c.getContext("2d");g.scale(dpr,dpr);const W=r.width,H=r.height,p=28,ym=12;
const x=k=>p+(W-2*p)*k/(D.length-1),y=v=>H-p-(H-2*p)*v/ym;g.clearRect(0,0,W,H);
g.strokeStyle="#ddd";g.fillStyle="#888";g.font="11px system-ui";[0,3,6,8,11].forEach(v=>{g.beginPath();g.moveTo(p,y(v));g.lineTo(W-p,y(v));g.stroke();g.fillText(v,4,y(v)+4)});
g.strokeStyle="#e8a317";g.setLineDash([4,3]);g.beginPath();g.moveTo(p,y(3));g.lineTo(W-p,y(3));g.stroke();g.setLineDash([]);
g.strokeStyle="#555";g.beginPath();let s=false;D.forEach((d,k)=>{if(k>i)return;if(d.uvi==null){s=false;return}s?g.lineTo(x(k),y(d.uvi)):g.moveTo(x(k),y(d.uvi));s=true});g.stroke();
D.forEach((d,k)=>{if(k>i)return;if(d.reminder){g.fillStyle="rgba(214,39,40,.18)";g.fillRect(x(k)-2,p,4,H-2*p)}
if(d.uvi!=null){g.fillStyle=COL[Math.max(d.pred_cat,0)];g.beginPath();g.arc(x(k),y(d.uvi),3,0,7);g.fill()}});
g.strokeStyle="#111";g.beginPath();g.moveTo(x(i),p);g.lineTo(x(i),H-p);g.stroke()}
function show(){const d=D[i];$("pos").value=i;$("clock").textContent=d.time;
$("uvi").textContent=d.uvi==null?"--":d.uvi.toFixed(2);const cc=d.cur_cat;
$("ccat").textContent=cc<0?"n/a":CATS[cc];$("ccat").style.background=cc<0?"#999":COL[cc];
$("fc").textContent=d.pred_cat<0?"--":CATS[d.pred_cat];$("fc").style.color=d.pred_cat<0?"#999":COL[d.pred_cat];$("fsrc").textContent=d.pred_source;
$("dose").textContent=Math.round(d.dose).toLocaleString();$("dq").textContent=d.data_status;$("sus").textContent="sustained >=3: "+d.sustained_min+" min";
const a=$("alert");if(d.reminder){a.className="on";a.textContent="Reminder ("+d.reminder_type+"): "+MSG[d.reminder_type]}
else{a.className="";a.textContent=d.reason?("No reminder - "+d.reason):"No reminder."}
const L=D.slice(0,i+1).filter(e=>e.reminder).reverse();$("log").innerHTML=L.map(e=>"<div><b>"+e.time.slice(11)+"</b> "+e.reminder_type+" (UVI "+e.uvi.toFixed(1)+")</div>").join("")||"<div>None yet</div>";draw()}
$("pos").oninput=e=>{i=+e.target.value;show()};
$("play").onclick=()=>{if(timer){clearInterval(timer);timer=null;$("play").textContent="Play";return}
if(i>=D.length-1)i=0;$("play").textContent="Pause";timer=setInterval(()=>{i++;show();if(i>=D.length-1){clearInterval(timer);timer=null;$("play").textContent="Play"}},400)};
addEventListener("resize",draw);show();
</script></body></html>"""


def write_html(tl, path):
    recs = []
    for _, r in tl.iterrows():
        recs.append({
            "time": r["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "uvi": None if not np.isfinite(r["uvi"]) else round(float(r["uvi"]), 2),
            "cur_cat": int(r["cur_cat"]), "pred_cat": int(r["pred_cat"]),
            "pred_source": r["pred_source"], "dose": float(r["dose_session"]),
            "data_status": r["data_status"], "sustained_min": int(r["sustained_min"]),
            "reminder": bool(r["reminder"]), "reminder_type": r["reminder_type"],
            "reason": r["reason"]})
    html = (HTML.replace("__DATA__", json.dumps(recs))
                .replace("__COLORS__", json.dumps(CAT_COLORS))
                .replace("__MSG__", json.dumps(MESSAGES)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    df, _ = a2.load_and_audit(data)
    feat, rf_params = add_forecasts(df)

    # Replay: step through intervals one at a time and time each step
    stream = feat[["timestamp", "uvi_now", "pred_cat"]].rename(columns={"uvi_now": "uvi"}).copy()
    lat = []
    for k in range(1, len(stream) + 1):
        t0 = time.perf_counter()
        reminder_engine(stream.iloc[:k])          # engine sees only data up to now
        lat.append((time.perf_counter() - t0) * 1000)
    tl = reminder_engine(stream)
    tl["cur_cat"] = a2.uvi_to_cat(tl["uvi"].fillna(-1))
    tl["pred_source"] = feat["pred_source"].values
    tl["dose_session"] = feat["dose_session"].values
    tl.to_csv(os.path.join(outdir, "aim3_replay_reminder_log.csv"), index=False)

    scen = run_scenarios()
    scen.to_csv(os.path.join(outdir, "aim3_scenario_tests.csv"), index=False)

    static_figure(tl, os.path.join(outdir, "fig2_reminder_replay.png"))
    write_html(tl, os.path.join(outdir, "sunsense_dashboard.html"))

    summary = {
        "WARNING": "Synthetic replay - software test only",
        "replay_intervals": int(len(tl)),
        "intervals_uvi_ge3": int((tl["uvi"] >= THRESH_UVI).sum()),
        "reminders_fired": int(tl["reminder"].sum()),
        "by_type": tl.loc[tl.reminder, "reminder_type"].value_counts().to_dict(),
        "suppressed_by_cooldown": int(tl["suppressed_by_cooldown"].sum()),
        "first_reminder": str(tl.loc[tl.reminder, "timestamp"].min()),
        "last_reminder": str(tl.loc[tl.reminder, "timestamp"].max()),
        "reminders_while_uvi_below_3": int((tl.reminder & (tl.uvi < THRESH_UVI)).sum()),
        "processing_latency_ms": {"median": round(float(np.median(lat)), 2),
                                  "p95": round(float(np.percentile(lat, 95)), 2),
                                  "max": round(float(np.max(lat)), 2)},
        "latency_note": "software processing time per replay step on this computer; "
                        "not BLE delivery or on-device display latency",
        "scenario_tests_passed": f"{int(scen['pass'].sum())}/{len(scen)}",
        "rf_params": rf_params,
    }
    with open(os.path.join(outdir, "aim3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(scen[["scenario", "expected", "observed_fire_minutes", "observed_types", "pass"]]
          .to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sunsense_synthetic_uv_dataset.csv")
    ap.add_argument("--out", default="results")
    a, _ = ap.parse_known_args()   # works in Jupyter too
    main(a.data, a.out)
