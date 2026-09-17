import pandas as pd

# The lists

causal_orders = {} # The causal tiers themselves (variables in tier N cannot be caused by variables in later tiers)
internally_forbidden_tiers = {} # Tiers whose variables cannot cause each other
protected_tiers = {} # Tiers whose variables cannot be caused, even by those that precede them in the causal order
required_edges = {}


# Trial 2

causal_orders[2] = [
    # Tier 0: Non-modifiable baseline demographics
    [
        "X_agegrp_0d", 
        "X_sex_0d"
        ],
    # Tier 1: Baseline symptoms & pre-existing comorbidities
    [
        "X_fever_0d",
        "X_cough_0d",
        "X_respdiff_0d",
        "X_diabetes_0d",
        "X_hyperten_0d",
        "X_comorb_0d",
    ],
    # Tier 2: Randomized intervention
    ["Treatment"],
    # Tier 3: Early post-treatment measurements
    ["YP_early_7d"],
    # Tier 4: Late post-treatment endpoints
    [
        "YP_recovery_time",
        "YP_late_12d",
        "YS_severity_worsen",
        "YS_pos_14d",
    ],
]

internally_forbidden_tiers[2] = [0]
protected_tiers[2] = [2]
required_edges[2] = []



# Trial 6

causal_orders[6] = [
    # Tier 0: Fixed Demographics & Prior Treatment History (Month 0)
    [
        "X_age_0m",
        "X_sex_0m",
        "X_statin_0m",
        "X_dm_treatment_0m",
    ],
    # Tier 1: Baseline Clinical Biomarkers & Vascular Parameters (Month 0)
    [
        "X_sbp_0m",
        "X_weight_0m",
        # Carotid Intima-Media Thickness
        "X_cca_imt_0m",
        "X_max_imt_0m",
        # Glycemic control
        "X_hba1c_0m",
        "X_hba1c_ancova_0m",
        "X_fbg_0m",
        "X_insulin_0m",
        "X_1_5ag_0m",
        # Renal / Kidney function
        "X_creatinine_0m",
        "X_albumin_0m",
        "X_egfr_0m",
        "X_cystatin_c_0m",
        # Lipid & Atherosclerosis markers
        "X_sdldl_0m",
        "X_adipo_0m",
        "X_rlpc_0m",
        "X_mda_ldl_0m",
    ],
    # Tier 2: Randomized Intervention
    [
        "Treatment",
    ],
    # Tier 3: Intermediate 12-Month Outcomes / Deltas
    [
        "pre_YP_delta_CCA_IMT_12m",
        "YS_delta_hba1c_12m",
        "YS_delta_fbg_12m",
        "YS_delta_insulin_12m",
        "YS_delta_1_5ag_12m",
        "YS_delta_weight_12m",
        "YS_delta_creatinine_12m",
        "YS_delta_albumin_12m",
        "YS_delta_egfr_12m",
        "YS_delta_cystatin_c_12m",
        "YS_delta_sdldl_12m",
        "YS_delta_adipo_12m",
        "YS_delta_rlpc_12m",
        "YS_delta_mda_ldl_12m",
    ],
    # Tier 4: Final 24-Month Outcomes / Deltas (Primary & Secondary Endpoints)
    [
        "YP_delta_CCA_IMT_24m",  # Primary study endpoint
        "YS_delta_hba1c_24m",
        "YS_delta_fbg_24m",
        "YS_delta_insulin_24m",
        "YS_delta_1_5ag_24m",
        "YS_delta_weight_24m",
        "YS_delta_creatinine_24m",
        "YS_delta_albumin_24m",
        "YS_delta_egfr_24m",
        "YS_delta_cystatin_c_24m",
        "YS_delta_sdldl_24m",
        "YS_delta_adipo_24m",
        "YS_delta_rlpc_24m",
        "YS_delta_mda_ldl_24m",
    ],
]

internally_forbidden_tiers[6] = [0]
protected_tiers[6] = [2]
required_edges[6] = []



# Trial 13

causal_orders[13] = [
    # Tier 0: Fixed Maternal Demographics & Obstetric History (0 min)
    [
        "X_AGE_0min",
        "X_PARITY_0min",
    ],
    # Tier 1: Baseline Intrapartum & Clinical Status (0 min)
    [
        "X_INDUCTION_0min",
        "X_HB_BASELINE_0min",
        "X_SI_BASELINE_0min",
    ],
    # Tier 2: Randomized Treatment Assignment
    [
        "Treatment",
    ],
    # Tier 3: 15-Minute Monitoring
    [
        "YS_SI_15min",
    ],
    # Tier 4: 30-Minute Monitoring (Hemodynamics & Blood Loss)
    [
        "YS_SYSBP_30min",
        "YS_DIASBP_30min",
        "YS_HR_30min",
        "YS_SI_30min",
        "YS_BLD_30min",
    ],
    # Tier 5: 45-Minute Monitoring
    [
        "YS_SI_45min",
    ],
    # Tier 6: 60-Minute Monitoring (Hemodynamics & Blood Loss)
    [
        "YS_SYSBP_60min",
        "YS_DIASBP_60min",
        "YS_HR_60min",
        "YS_SI_60min",
        "YS_BLD_60min",
    ],
    # Tier 7: Cumulative & Final Endpoints
    [
        "YP_TOTALBLOOD",
        "YP_PPH500",
        "YS_PPH1000",
        "YS_UTEROTONICOS",
        "YS_HB_CHG",
    ],
]

internally_forbidden_tiers[13] = [0]
protected_tiers[13] = [2]
required_edges[13] = []



# Trial 29

causal_orders[29] = [
    # Tier 0: Demographics and Pre-Existing Clinical History (Week 0)
    [
        "X_age_0w",
        "X_sex_0w",
        "X_study_0w",
        "X_stage_0w",
        "X_status_0w",
        "X_on_art_0w",
        "X_on_tb_meds_0w",
    ],
    # Tier 1: Baseline Neurological, Systemic, and CSF Laboratory Measures (Week 0)
    [
        "X_cd4_0w",
        "X_hiv_rna_0w",
        "X_time_hiv_rna_0w",
        "X_gcs_0w",
        "X_gcs_normal_0w",
        "X_hemoglobin_0w",
        "X_creatinine_mg_0w",
        "X_creatinine_umol_0w",
        "X_csf_qcc_0w",
        "X_csf_wbc_0w",
        "X_csf_wbc_low_0w",
        "X_csf_opening_pressure_0w",
        "X_csf_protein_0w",
        "X_csf_sterile_baseline_0w",
    ],
    # Tier 2: Randomized Treatment Assignment
    [
        "Treatment",
    ],
    # Tier 3: Early Post-Treatment CSF Sterility (Day 14)
    [
        "YS_sterile_14d",
    ],
    # Tier 4: Week 4 Intermediate Follow-up Assessments
    [
        "YS_depression_4w",
        "YS_visit_4w_timing",
    ],
    # Tier 5: Week 12 Intermediate Follow-up Assessments
    [
        "YS_depression_12w",
        "YS_visit_12w_timing",
    ],
    # Tier 6: Final 18-Week Mortality Endpoints and Cumulative Procedures
    [
        "YS_sterile_18w",
        "YS_num_lumbar_punctures",
        "YP_survival_18w",
        "YP_death",
    ],
]

internally_forbidden_tiers[29] = [0]
protected_tiers[29] = [2]
required_edges[29] = []



# Trial 37

causal_orders[37] = [
    # Tier 0: Pre-Procedural Demographics, Risk Score, and Clinical History (Day 0)
    [
        "X_site_0d",
        "X_age_0d",
        "X_gender_0d",
        "X_risk_score_0d",
        "X_status_0d",
        "X_pep_0d",
        "X_recpanc_0d",
        "X_psphinc_0d",
        "X_soc_0d",
        "X_chole_0d",
        "X_pbmal_0d",
        "X_asa81_0d",
        "X_asa325_0d",
        "X_asa_any_0d",
    ],
    # Tier 1: Intra-Procedural ERCP Characteristics and Interventions (Day 0)
    [
        "X_trainee_0d",
        "X_difcan_0d",
        "X_precut_0d",
        "X_pneudil_0d",
        "X_amp_0d",
        "X_paninj_0d",
        "X_acinar_0d",
        "X_brush_0d",
        "X_sodsom_0d",
        "X_bsphinc_0d",
        "X_bstent_0d",
        "X_prophystent_0d",
        "X_therastent_0d",
        "X_pdstent_0d",
    ],
    # Tier 2: Randomized Treatment Assignment
    [
        "Treatment",
    ],
    # Tier 3: Post-ERCP Pancreatitis Primary Outcome (Day 5)
    [
        "YP_pep_5d",
    ],
]

internally_forbidden_tiers[37] = [0]
protected_tiers[37] = [2]
required_edges[37] = []



# Trial 108

causal_orders[108] = [
    # Tier 0: Fixed Demographics and Study Center
    [
        "X_hospital",
        "X_age_years",
        "X_sex",
    ],
    # Tier 1: Baseline Renal Function and Organ Failure Severity (Day 0)
    [
        "X_baseline_creatinine",
        "X_initial_egfr",
        "X_sofa_0d",
    ],
    # Tier 2: Randomized Treatment Assignment
    [
        "Treatment",
    ],
    # Tier 3: 14-Day Component Clinical Endpoints
    [
        "YS_death_14d",
        "YS_dialysis_14d",
        "YS_aki_progression_14d",
    ],
    # Tier 4: 14-Day Prespecified Primary Composite Endpoint
    [
        "YP_composite_14d",
    ],
]

internally_forbidden_tiers[108] = [0]
protected_tiers[108] = [2]
required_edges[108] = []



# Trial 116

causal_orders[116] = [
    # Tier 0: Demographics and Socioeconomic Background
    [
        "X_source_id",
        "X_age_years",
        "X_education_code",
        "X_religion_code",
        "X_residence_place_code",
        "X_job_code",
        "X_marital_status_code",
        "X_income_code",
    ],
    # Tier 1: Clinical Oncology History and Baseline Quality of Life / Psychological Scores
    [
        "X_cancer_diagnosis_code",
        "X_cancer_stage_code",
        "X_treatment_type_code",
        "X_chemotherapy_time",
        "X_regimen_number",
        "X_dass_total_0m",
        "X_factg_total_0m",
    ],
    # Tier 2: Randomized Treatment Assignment
    [
        "Treatment",
    ],
    # Tier 3: Intermediate Follow-up at Assessment T1
    [
        "YS_dass_total_t1",
        "YS_delta_dass_total_t1",
    ],
    # Tier 4: Final Follow-up at Assessment T2 (Primary and Secondary Endpoints)
    [
        "YP_dass_total_t2",
        "YP_delta_dass_total_t2",
        "YP_factg_total_t2",
        "YP_delta_factg_total_t2",
        "YS_dass_stress_t2",
        "YS_dass_anxiety_t2",
        "YS_dass_depression_t2",
    ],
]

internally_forbidden_tiers[116] = [0]
protected_tiers[116] = [2]
required_edges[116] = []



# Trial 120

causal_orders[120] = [
    # Tier 0: Baseline Patient Demographics and Tumor Staging
    [
        "X_source_subject_id",
        "X_melanoma_stage",
    ],
    # Tier 1: Randomized Treatment Assignment
    [
        "Treatment",
    ],
    # Tier 2: Progression-Free Survival Endpoints
    [
        "YP_progression_free_survival_months",
        "YP_pfs_event",
    ],
    # Tier 3: Overall Survival Endpoints
    [
        "YS_overall_survival_months",
        "YS_overall_survival_event",
    ],
]

internally_forbidden_tiers[120] = [0]
protected_tiers[120] = [2]
required_edges[120] = []



# Uniting all dictionaries into a dataframe

df_background_knowledge = pd.DataFrame()
df_background_knowledge['causal_orders'] = causal_orders
df_background_knowledge['internally_forbidden_tiers'] = internally_forbidden_tiers
df_background_knowledge['protected_tiers'] = protected_tiers
df_background_knowledge['required_edges'] = required_edges