SELECTED_FEATURES = [
    'year',
    'number of affected users',
    'incident resolution time (in hours)',
    'data breach in gb',
    'attack type_ddos',
    'attack type_malware',
    'attack type_man-in-the-middle',
    'attack type_phishing',
    'attack type_ransomware',
    'attack type_sql injection',
    'target industry_banking',
    'target industry_education',
    'target industry_government',
    'target industry_healthcare',
    'target industry_it',
    'target industry_retail',
    'target industry_telecommunications',
    'security vulnerability type_social engineering',
    'security vulnerability type_unpatched software',
    'security vulnerability type_weak passwords',
    'security vulnerability type_zero day',
    'defense mechanism used_ai based detection',
    'defense mechanism used_antivirus',
    'defense mechanism used_encryption',
    'defense mechanism used_firewall',
    'defense mechanism used_vpn',
]

MODEL_SAVE_PATH = 'models/severity_model.pkl'
TEST_SIZE = 0.3
RANDOM_STATE = 42
