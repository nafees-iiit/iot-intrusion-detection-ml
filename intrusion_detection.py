"""
Machine-Learning Intrusion Detection for IoT Gateways (NSL-KDD benchmark)
=======================================================================
Author : Mohammad Nafees (Ph.D. Scholar, ECE, MNNIT Allahabad)

Problem
-------
IoT devices are weak at protecting themselves, so attacks are usually caught at
the gateway/edge by an Intrusion Detection System (IDS) that inspects network
connections. Each connection is described by 41 features (duration, protocol,
bytes sent, error rates, ...). The IDS must decide: NORMAL or ATTACK, and if
attack, which family (DoS, Probe, R2L, U2R).

Data
----
NSL-KDD, a standard public benchmark (cleaned version of KDD Cup 99).
Training file : KDDTrain+  (125,973 connections)
Test file     : KDDTest+   (22,544 connections, includes 17 attack types that
                            NEVER appear in training -> a realistic test)

Note: NSL-KDD is a general network-traffic benchmark, widely used to develop
lightweight IDS models for IoT gateways; it is not captured from IoT devices.
"""

import os
import time
import pickle
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

SEED = 42
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------------
# 1. Download and load the dataset
# ------------------------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
for fname in ["KDDTrain+.txt", "KDDTest+.txt"]:
    if not os.path.exists(f"data/{fname}"):
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(BASE_URL + fname, f"data/{fname}")

COLUMNS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
           "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
           "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]
train = pd.read_csv("data/KDDTrain+.txt", names=COLUMNS)
test = pd.read_csv("data/KDDTest+.txt", names=COLUMNS)

# ------------------------------------------------------------------
# 2. Labels: attack name -> family (standard NSL-KDD mapping)
# ------------------------------------------------------------------
FAMILY = {
    "DoS":   ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb",
              "processtable", "udpstorm"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L":   ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient",
              "warezmaster", "named", "sendmail", "snmpgetattack", "snmpguess", "worm",
              "xlock", "xsnoop"],
    "U2R":   ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps",
              "sqlattack", "xterm"],
}
name_to_family = {a: fam for fam, attacks in FAMILY.items() for a in attacks}
for df in (train, test):
    df["family"] = df["label"].map(lambda x: "Normal" if x == "normal" else name_to_family[x])
    df["is_attack"] = (df["label"] != "normal").astype(int)

new_attacks = sorted(set(test["label"]) - set(train["label"]))
test["new_attack"] = test["label"].isin(new_attacks)
print(f"Train: {len(train)} connections | Test: {len(test)} connections")
print("Train classes:", train["family"].value_counts().to_dict())
print("Test  classes:", test["family"].value_counts().to_dict())
print(f"{len(new_attacks)} attack types appear ONLY in the test set: {', '.join(new_attacks)}")

# ------------------------------------------------------------------
# 3. Pre-processing: one-hot encode text columns, scale numbers
# ------------------------------------------------------------------
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS[:-2] if c not in CATEGORICAL]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])
X_train, y_train = train[CATEGORICAL + NUMERIC], train["is_attack"].values
X_test, y_test = test[CATEGORICAL + NUMERIC], test["is_attack"].values
# 20 % of the training file is held out: it contains only KNOWN attack types
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                            random_state=SEED, stratify=y_train)

# ------------------------------------------------------------------
# 4. Binary detection (Normal vs Attack) with four models
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree (depth 12)": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                          early_stopping=True, random_state=SEED),
}

def evaluate(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall (detection rate)": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "False alarm rate": fp / (fp + tn),
            "ROC-AUC": roc_auc_score(y_true, y_score)}

rows, fitted, roc_data = [], {}, {}
for name, clf in models.items():
    pipe = Pipeline([("prep", clone(preprocess)), ("clf", clf)])   # own copy per model
    t0 = time.time(); pipe.fit(X_tr, y_tr); fit_t = time.time() - t0
    fitted[name] = pipe
    for split, X, y in [("Validation (known attacks)", X_val, y_val),
                        ("KDDTest+ (incl. new attacks)", X_test, y_test)]:
        score = pipe.predict_proba(X)[:, 1]
        pred = (score >= 0.5).astype(int)
        rows.append({"Model": name, "Data": split, **evaluate(y, pred, score)})
        if split.startswith("KDDTest"):
            roc_data[name] = roc_curve(y, score)
    t0 = time.time(); pipe.predict(X_test); inf_t = (time.time() - t0) / len(X_test)
    size_kb = len(pickle.dumps(pipe)) / 1024
    rows[-1].update({"Train time (s)": fit_t, "Inference (us/connection)": 1e6 * inf_t,
                     "Model size (KB)": size_kb})

res = pd.DataFrame(rows)
res.to_csv("results/binary_detection_results.csv", index=False)
pd.set_option("display.width", 200)
show = ["Model", "Data", "Accuracy", "Precision", "Recall (detection rate)", "F1",
        "False alarm rate", "ROC-AUC"]
print("\nBINARY DETECTION (Normal vs Attack):")
print(res[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nEdge-deployment cost (measured on this computer):")
print(res.dropna(subset=["Model size (KB)"])[["Model", "Train time (s)", "Inference (us/connection)",
      "Model size (KB)"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# Detection of NEW vs KNOWN attack types on the test set (Random Forest)
rf = fitted["Random Forest"]
test_pred = rf.predict(X_test)
att = test["is_attack"] == 1
known_dr = test_pred[att & ~test["new_attack"]].mean()
new_dr = test_pred[att & test["new_attack"]].mean()
print(f"\nRandom Forest detection rate on test attacks: KNOWN types = {known_dr*100:.1f} %, "
      f"NEW (never seen) types = {new_dr*100:.1f} %")

# ROC analysis (discussion only -- the threshold is NOT tuned on the test set)
fpr, tpr, _ = roc_data["Random Forest"]
for limit in (0.05, 0.10):
    print(f"Random Forest ROC: if a false-alarm rate of {limit*100:.0f} % is acceptable, "
          f"detection rate = {tpr[fpr <= limit].max()*100:.1f} %")

# ------------------------------------------------------------------
# 5. Multi-class: which attack family? (Random Forest)
# ------------------------------------------------------------------
families = ["Normal", "DoS", "Probe", "R2L", "U2R"]
mc = Pipeline([("prep", clone(preprocess)),
               ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED))])
mc.fit(X_train, train["family"])
fam_pred = mc.predict(X_test)
cm5 = confusion_matrix(test["family"], fam_pred, labels=families)
per_class = pd.DataFrame({"Family": families,
                          "Test samples": cm5.sum(axis=1),
                          "Recall": np.diag(cm5) / cm5.sum(axis=1)})
per_class.to_csv("results/multiclass_per_family_recall.csv", index=False)
print("\nMULTI-CLASS (Random Forest) on KDDTest+: accuracy = "
      f"{accuracy_score(test['family'], fam_pred)*100:.2f} %")
print(per_class.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ------------------------------------------------------------------
# 6. Figures
# ------------------------------------------------------------------
# (a) F1: validation vs test for every model
piv = res.pivot(index="Model", columns="Data", values="F1").loc[list(models)]
piv = piv[["Validation (known attacks)", "KDDTest+ (incl. new attacks)"]]
ax = piv.plot(kind="bar", figsize=(8, 4.6), color=["tab:green", "tab:red"], rot=0)
ax.legend(loc="lower right", fontsize=8)
ax.set_ylabel("F1-score"); ax.set_ylim(0.5, 1.02); ax.grid(axis="y", alpha=0.3)
ax.set_title("Known attacks (validation) vs new attacks included (KDDTest+)")
ax.set_xticklabels([t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()], fontsize=8)
for c in ax.containers: ax.bar_label(c, fmt="%.3f", fontsize=7)
plt.tight_layout(); plt.savefig("results/f1_validation_vs_test.png", dpi=200); plt.close()

# (b) ROC curves on KDDTest+
plt.figure(figsize=(5.8, 5.2))
for name, (fpr, tpr, _) in roc_data.items():
    auc = res[(res.Model == name) & res.Data.str.startswith("KDDTest")]["ROC-AUC"].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False alarm rate (FPR)"); plt.ylabel("Detection rate (TPR)")
plt.title("ROC curves on KDDTest+"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/roc_curves.png", dpi=200); plt.close()

# (c) Confusion matrix, binary, Random Forest on KDDTest+
cm2 = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(4.8, 4.2)); plt.imshow(cm2, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm2[i, j]:,}", ha="center", va="center", fontsize=12,
                 color="white" if cm2[i, j] > cm2.max() / 2 else "black")
plt.xticks([0, 1], ["Normal", "Attack"]); plt.yticks([0, 1], ["Normal", "Attack"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Random Forest on KDDTest+")
plt.tight_layout(); plt.savefig("results/confusion_matrix_binary.png", dpi=200); plt.close()

# (d) Multi-class confusion matrix (row-normalised)
cmn = cm5 / cm5.sum(axis=1, keepdims=True)
plt.figure(figsize=(6, 5)); plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1); plt.colorbar()
for i in range(5):
    for j in range(5):
        plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                 color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
plt.xticks(range(5), families); plt.yticks(range(5), families)
plt.xlabel("Predicted family"); plt.ylabel("True family")
plt.title("Attack families on KDDTest+ (row-normalised)")
plt.tight_layout(); plt.savefig("results/confusion_matrix_multiclass.png", dpi=200); plt.close()

# (e) Top-15 feature importances (Random Forest, binary)
names = list(rf.named_steps["prep"].get_feature_names_out())
names = [n.replace("cat__", "").replace("num__", "") for n in names]
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).nlargest(15)[::-1]
plt.figure(figsize=(7, 5)); plt.barh(imp.index, imp.values, color="tab:blue")
plt.xlabel("Importance"); plt.title("Top-15 features used by the Random Forest")
plt.tight_layout(); plt.savefig("results/feature_importance.png", dpi=200); plt.close()
imp[::-1].to_csv("results/top15_features.csv", header=["importance"])
print("\nTop-5 features:", ", ".join(imp[::-1].index[:5]))
print("\nAll results saved in the 'results' folder.")
