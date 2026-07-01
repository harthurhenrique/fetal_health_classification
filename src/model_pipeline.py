import joblib
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, recall_score, 
                             precision_score, roc_auc_score)

flag_pipeline_treino = 1
#0: Pipeline do Naive Bayes
#1: Pipeline do SVM e Decision tree

df_base = pd.read_csv('src/data/fetal_health.csv')
df_base = df_base.drop(columns=['prolongued_decelerations', 'severe_decelerations', 'histogram_number_of_zeroes'])


X = df_base.drop('fetal_health', axis=1)
y  = df_base['fetal_health']

# Definição dos Pipelines
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
    # "Naive Bayes": Pipeline([
    #     ('model', GaussianNB()) 
    # ])

}

# Configuração da Validação Cruzada
# 10 folds garantem que testaremos o modelo em 10 divisões diferentes do dataset
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


metrics = {
    'Acurácia': 'accuracy',
    'F1-Score (Weighted)': 'f1_weighted',
    'AUC-ROC': 'roc_auc_ovr',
    'Precisão': 'precision_weighted',
    'Recall': 'recall_weighted'
}

resumo_geral = []

# ExcelWriter para salvar os detalhes de cada fold em abas separadas
writer_detalhes = pd.ExcelWriter('detalhes_execucao_folds.xlsx', engine='openpyxl')

if flag_pipeline_treino == 1:

    for name, pipeline in pipelines.items():    
        print(f"Processando {name}...")
        
        start_time = time.time()
        # Executa a validação e transforma o dicionário integral em DataFrame
        scores = cross_validate(pipeline, X, y, cv=skf, scoring=metrics, n_jobs=-1)
        end_time = time.time()
        
        # DataFrame com os resultados de cada um dos 10 folds
        df_folds = pd.DataFrame(scores)
        df_folds.insert(0, 'Execucao_Fold', range(1, 11))
        
        # Salva os detalhes deste modelo em uma aba específica do arquivo de detalhes
        df_folds.to_excel(writer_detalhes, sheet_name=name, index=False)
        
        # Calcula a média para o Resumo Geral
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

elif flag_pipeline_treino == 0:

    # Definição das estratégias de balanceamento
    pipeline_nb = pipelines['Naive Bayes']

    estrategias = {
        "Naive Bayes (Oversampling)": SMOTE(random_state=42),
        "Naive Bayes (Undersampling)": RandomUnderSampler(random_state=42)
    }

    for name, sampler in estrategias.items():
        print(f"Processando {name}...")
        
        start_time = time.time()
        
        fold_results = []
        y_true_all = []
        y_pred_all = []
        
        # Loop manual para garantir balanceamento apenas no treino (Stratified)
        for i, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            # Separação dos dados
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # --- APLICAÇÃO DO BALANCEAMENTO ---
            # Aplica a técnica (SMOTE ou Under) apenas nos dados de treino do fold
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            
            # Treino e Predição
            t0_fit = time.time()
            pipeline_nb.fit(X_res, y_res)
            preds = pipeline_nb.predict(X_test)
            t1_fit = time.time()
            
            preds = pipeline_nb.predict(X_test)
            probs = pipeline_nb.predict_proba(X_test) 

            # Armazena para Matriz de Confusão e Métricas
            y_true_all.extend(y_test)
            y_pred_all.extend(preds)
            
            # Simula o retorno do cross_validate 
            fold_results.append({
                'Execucao_Fold': i,
                'fit_time': t1_fit - t0_fit,
                'test_Acurácia': accuracy_score(y_test, preds),
                'test_F1-Score (Weighted)': f1_score(y_test, preds, average='weighted'),
                'test_AUC-ROC': roc_auc_score(y_test, probs, multi_class='ovr', average='weighted'),
                'test_Precisão': precision_score(y_test, preds, average='weighted'),
                'test_Recall': recall_score(y_test, preds, average='weighted')
            })

        end_time = time.time()
        
        #  DataFrame com resultados dos Folds 
        df_folds = pd.DataFrame(fold_results)
        df_folds.to_excel(writer_detalhes, sheet_name=name, index=False)
        
        # --- 2. Resumo Geral ---
        df_mean = df_folds.drop(columns=['Execucao_Fold']).mean().to_frame().T
        df_mean.insert(0, 'Modelo', name)
        df_mean['Tempo_Total_Seg'] = round(end_time - start_time, 4)
        resumo_geral.append(df_mean)

        # --- 3. Matriz de Confusão ---
        cm = confusion_matrix(y_true_all, y_pred_all)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {name}')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        
        nome_arquivo = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f'src/plots/matriz_confusao_{nome_arquivo}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Matriz de confusão salva: matriz_confusao_{nome_arquivo}.png")

        # --- 4. Salvar Modelo Final e Parquet ---
        # Para o modelo final, balanceamos a base completa
        X_final_res, y_final_res = sampler.fit_resample(X, y)
        pipeline_nb.fit(X_final_res, y_final_res)
        
        joblib.dump(pipeline_nb, f"modelo_{nome_arquivo}.joblib")
        df_folds.to_parquet(f"execucao_completa_{nome_arquivo}.parquet", index=False)


else:
    print('Nenhum pipeline selecionado para treinamento. Verifique a configuração da flag_pipeline_treino.')

writer_detalhes.close()

# Consolida e salva o Resumo Geral
df_resumo_final = pd.concat(resumo_geral, ignore_index=True)
df_resumo_final.to_excel("resultados_gerais_media.xlsx", index=False)

print("\nArquivos gerados com sucesso:")
print("- detalhes_execucao_folds.xlsx (Resultados de cada fold)")
print("- resultados_gerais_media.xlsx (Média consolidada)")
print("- Arquivos .parquet e .joblib para cada modelo")