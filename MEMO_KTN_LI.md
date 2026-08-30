# MÉMO KTN:Li COMPLET — PROJET RATISS × TISSAGE SPONTANÉ 3D

> **Pour la prochaine session.** Ce fichier est la mémoire complète du projet
> KTN:Li. Lis-le AVANT de toucher à quoi que ce soit.

## CONTEXTE — le papier réel (vérifié)

Jonathan a lié le moteur RATISS à la découverte du **tissage spontané 3D dans
KTN:Li** (ferroélectrique). Le papier existe et a été vérifié via l'API PubMed :

> **"Spontaneous formation and optical manipulation of a woven domain fabric
> in a ferroelectric crystal"**
> - Xin F., Gelkop Y., van der Veer E., Noheda B., Falsi L., Zhang G., Bo F.,
>   Agranat A.J., DelRe E. (Nankai / Sapienza Rome / Hebrew U / Groningen)
> - *Light: Science & Applications* **15**, 315 (14 juillet 2026)
> - DOI: 10.1038/s41377-026-02374-7 · PMCID: PMC13370020 (open access)
> - Full text PMC téléchargé : "topologically-protected defect", "braided
>   domain structure", domain walls chargés, manipulation optique
>   site-par-site au laser visible, régénération à chaque cycle thermique,
>   comparaison ADN/quasi-cristaux (Shechtman)

## CE QUI EST FAIT — LES 5 PHASES LIVRÉES

### Phase 1 — Données synthétiques KTN:Li ✓
- `ratiss_photoinduced/ktn_woven.py` : `make_woven()` (9 brins hélicoïdaux
  entrelacés, 900 pts) vs `make_aligned()` (domaines parallèles),
  `optical_unweave()` (perturbation laser 514 nm gaussienne),
  `thermal_regenerate()` (nouveau motif à chaque seed)

### Phase 2 — Caractérisation topologique ✓ (SUCCÈS)
- **P_sig(woven) = 0.677 vs P_sig(aligned) = 0.231 → ratio 2.9×**
- Le tissage est détectable par persistance Vietoris-Rips GF(2) (H1)
- 80 points sous-échantillonnés (limite de l'engine Rips sur triangles)

### Phase 3 — Réécriture optique ✓ (résultat négatif informatif)
- Perturbation laser gaussienne locale : P_sig quasi inchangé (1.00×)
- **Le tissage est résilient à la perturbation locale** — cohérent avec la
  protection topologique rapportée dans le papier

### Phase 4 — Régénération thermique ✓
- `ktn_phase4_hysteresis.py` : cycles thermiques sur le tissage
- **Invariance de régénération : drift 1.8%** (P_sig 0.789 → 0.803)
- Le motif change à chaque cycle, la signature topologique reste — analogue
  direct du "nouveau motif à chaque refroidissement" du papier
- Aire d'hystérésis mesurée vs vitesse (α≈0 ici car modèle d'interpolation
  statique — limitation honnête documentée)

### Phase 5 — Validation QPU IBM ✓ (avec itérations)
- `ktn_phase5_qpu.py` : état "woven" (superposition uniforme des 6 configs à
  2 fermions, C_ij = 1/3) vs état "aligned" (|0011>)
- **Itération 1 (bug)** : post-sélection sur toutes les bases biaisait XX/YY
  (les rotations H changent le nombre de particules) → corrigée sur Z seul
- **Itération 2 (bug)** : référence exacte woven C_ij = 1/6 → corrigée à 1/3
- **Itération 3 (backend)** : ibm_fez saturé (573 jobs) → relance marrakesh
- **Job fez `da81b5m0ukec7383sf20` DONE** : woven S=0.182 (exact 0.267),
  edge=0.265 (exact 0.333) — la structure tissée survit au hardware
- **Contraste QPU woven vs aligned : +0.260 positif**
- Artefacts : `artifacts/ktn_phase5_qpu.json`, counts sauvegardés

### Phase 6 — Préprint ✓
- `preprint/preprint_ktn.tex` compilé → `preprint_ktn.pdf` (8 pages)
- 4 figures KTN : figK1 textures 3D, figK2 diagrammes persistance,
  figK3 régénération, figK4 QPU hardware
- Citation réelle Xin et al. 2026 + Shechtman 1984
- Section limites honnêtes (textures synthétiques, proxy QPU, H1 seulement)
- 0 référence cassée (vérifié par extraction texte)

## ÉTAT DES TESTS
**24/24 verts** (`pytest tests/ -q`), dont 3 nouveaux tests KTN :
`test_woven_detected_over_aligned`, `test_regeneration_invariance`,
`test_woven_exact_correlation_structure`

## ACCÈS IBM QUANTUM (procédure qui marche)
1. Lire la clé : fichier `clef` dans le repo (format "ma cle ibm XXXX")
2. Token IAM : POST https://iam.cloud.ibm.com/identity/token avec
   grant_type=urn:ibm:params:oauth:grant-type:apikey
3. CRN instance : GET
   https://resource-controller.cloud.ibm.com/v2/resource_instances?type=service_instance
   → `crn:v1:bluemix:public:quantum-computing:us-east:a/16793dc4...`
4. `QiskitRuntimeService(channel="ibm_quantum_platform", token=..., instance=CRN)`
5. **Ne PAS utiliser** `instance="open-instance"` (rejeté) — utiliser le CRN
6. **Ne PAS utiliser** `backend.run()` (déprécié) — utiliser
   `SamplerV2(mode=backend)` + `job.result()[i].data.meas.get_counts()`
7. Backends : ibm_fez (souvent saturé, 573 jobs), ibm_marrakesh (20),
   ibm_kingston (41) — vérifier `backend.status().pending_jobs` avant

## TRAVAIL PRÉCÉDENT (inchangé, tout valide)
Le pipeline SSH complet (sweep, rampe, hystérésis α=1.05, Lindblad, QPU
marrakesh/fez) est documenté dans README.md et le préprint SSH
(`preprint/preprint.pdf`, 12 pages). 21 tests de base + 3 KTN.

## CE QUI RESTE À FAIRE (prochaines sessions)
1. **Données réelles KTN:Li** : contacter DelRe/Noheda pour les données
   d'imagerie 3D (SHG microscopy) ou digitaliser les figures PMC13370020
   → injecter dans le pipeline tel quel (Phase 1 remplace le synthétique)
2. **Job aligned marrakesh** `da81r1k6l22c73do6msg` : encore QUEUED à la
   fin de session — récupérer pour compléter le contraste woven/aligned sur
   marrakesh (fez déjà fait)
3. **H2 (cavités)** : étendre l'engine Rips aux tétraèdres pour capter les
   volumes clos du tissage 3D (le vrai contenu 3D du tissage)
4. **Soumission arXiv** : catégorie cond-mat.mtrl-sci ou quant-ph
   (le PDF est prêt)
5. **Mettre à jour MEMO_GLOBAL.md** du repo Ratiss-experimental-IA- avec le
   lien vers ce projet

## RÈGLES DE LA SESSION (conseil de Jonathan)
- **Jamais figé, toujours itérer** : si un circuit casse au QPU, remplacer
  les circuits auxiliaires et relancer. C'est comme ça qu'on est passé de
  "P_sig inversé" à "contraste validé".
- Ne jamais prétendre un résultat non mesuré — documenter les échecs.
- Repo `Travaux` = privé (contient `clef`) — ne pas le rendre public.
