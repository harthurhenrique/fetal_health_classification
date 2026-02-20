import joblib
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


df_base = pd.read_csv('src/data/fetal_health.csv')

X = df_base.drop('fetal_health', axis=1)
y  = df_base['fetal_health']

# 1. Definição dos Pipelines
# Mantemos o StandardScaler apenas para o SVM, pois árvores não precisam de normalização
pipelines = {
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
    ]),
    "Decision Tree (Gini)": Pipeline([
        ('model', DecisionTreeClassifier(criterion='gini', class_weight='balanced', random_state=42))
    ]),
    "Decision Tree (Entropy)": Pipeline([
        ('model', DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42))
    ])

}

# 2. Configuração da Validação Cruzada
# 10 folds garantem que testaremos o modelo em 10 divisões diferentes do dataset
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 3. Métricas de avaliação que queremos observar
metrics = {
    'Acurácia': 'accuracy',
    'F1-Score (Weighted)': 'f1_weighted',
    'AUC-ROC': 'roc_auc_ovr',
    'Precisão': 'precision_weighted',
    'Recall': 'recall_weighted'
}

resumo_geral = []

# Criamos um ExcelWriter para salvar os detalhes de cada fold em abas separadas
writer_detalhes = pd.ExcelWriter('detalhes_execucao_folds.xlsx', engine='openpyxl')

for name, pipeline in pipelines.items():    

    print(f"Processando {name}...")
    
    start_time = time.time()
    # 1. Executa a validação e transforma o dicionário integral em DataFrame
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=metrics, n_jobs=-1)
    end_time = time.time()
    
    # Criamos o DataFrame com os resultados de cada um dos 10 folds
    df_folds = pd.DataFrame(scores)
    df_folds.insert(0, 'Execucao_Fold', range(1, 11))
    
    # Salva os detalhes deste modelo em uma aba específica do arquivo de detalhes
    df_folds.to_excel(writer_detalhes, sheet_name=name, index=False)
    
    # 2. Calcula a média para o Resumo Geral
    df_mean = df_folds.drop(columns=['Execucao_Fold']).mean().to_frame().T
    df_mean.insert(0, 'Modelo', name)
    df_mean['Tempo_Total_Seg'] = round(end_time - start_time, 4)
    
    resumo_geral.append(df_mean)

    # --- Matriz de Confusão ---
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {name}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    # Salva a imagem na pasta do projeto
    plt.savefig(f'src/plots/matriz_confusao_{name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close() # Fecha a figura para não consumir memória
    print(f"  -> Matriz de confusão salva: matriz_confusao_{name.lower()}.png")


    # --- Salvar Modelo e Parquet individual ---
    pipeline.fit(X, y)
    joblib.dump(pipeline, f"modelo_{name.lower()}.joblib")
    df_folds.to_parquet(f"execucao_completa_{name.lower()}.parquet", index=False)

# Fecha o arquivo de detalhes
writer_detalhes.close()

# 3. Consolida e salva o Resumo Geral
df_resumo_final = pd.concat(resumo_geral, ignore_index=True)
df_resumo_final.to_excel("resultados_gerais_media.xlsx", index=False)

print("\nArquivos gerados com sucesso:")
print("- detalhes_execucao_folds.xlsx (Resultados de cada fold)")
print("- resultados_gerais_media.xlsx (Média consolidada)")
print("- Arquivos .parquet e .joblib para cada modelo")