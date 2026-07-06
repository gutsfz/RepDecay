# RepDecay: Estimativa de Proximidade da Falha no Agachamento via Visão Computacional

## 1. O problema real que o projeto resolve

Treino baseado em velocidade (VBT) é um método com respaldo científico real pra gerenciar fadiga e estimar quantas repetições em reserva (RIR) um atleta ainda tem numa série, sem precisar levar toda série até a falha. O problema é o acesso a essa tecnologia.

O hardware validado (transdutores lineares de posição) é caro: RepOne custa 399 dólares, Vitruve 447, Tendo 1.329, e o GymAware (considerado padrão-ouro) quase 2.000 dólares, muitas vezes com assinatura anual em cima disso. Isso é uma barreira significativa, e ainda mais pesada num contexto como o brasileiro, onde esse valor em dólar representa uma fatia bem maior de renda.

A alternativa gratuita ou barata são apps de celular, mas a confiabilidade deles é questionável. Um estudo publicado em 2024 comparando três apps (Qwik VBT, Metric VBT, MyLift) contra um transdutor validado como referência encontrou, em 589 repetições testadas, que o Metric VBT não identificou 52 delas e o MyLift não identificou 175 (quase 30% das repetições simplesmente não foram contadas). Só o Qwik VBT chegou a um nível de precisão comparável ao hardware validado.

Existe, portanto, uma lacuna real: não existe uma opção gratuita, totalmente automática (sem exigir que o usuário marque a barra manualmente quadro a quadro) e minimamente confiável pra estimar proximidade da falha via vídeo. O RepDecay se propõe a explorar essa lacuna usando ferramentas 100% gratuitas.

## 2. Fundamentação científica

### O que já é cientificamente estabelecido

- A relação entre queda de velocidade e RIR é real e documentada. Um estudo de 2025 publicado na PeerJ (Paulsen et al.), com 19 atletas treinados e quase 3.000 medições ao longo de 6 semanas de treino real, encontrou correlação entre velocidade média da barra e RIR percebido variando de 0,3 a 0,9 entre indivíduos no agachamento, com média de 0,6. Isso confirma duas coisas que orientam o projeto: a relação existe, mas varia bastante de pessoa pra pessoa, o que justifica calibração individual em vez de um modelo único genérico.
- O mesmo estudo indica que treino de hipertrofia costuma se beneficiar de limiares de perda de velocidade mais altos (acima de 25%, perto da falha), enquanto força e potência usam limiares mais baixos. Isso valida o foco do projeto em detectar proximidade da falha especificamente.
- Existem sistemas de visão computacional para rastreamento automático de barra (não landmark corporal) já validados cientificamente contra transdutores lineares, mostrando que a abordagem geral "vídeo em vez de sensor físico" é tecnicamente viável quando bem implementada.
- Métodos de pose estimation sem marcador (OpenPose, DeepLabCut, e o próprio MediaPipe, citado em estudos de biomecânica esportiva) são usados em pesquisa científica de cinemática, com margem de erro conhecida e documentada: diferenças sistemáticas de 30 a 50mm na localização de quadril e joelho comparado a sistemas de captura por marcador, e erro de amplitude de movimento articular entre 2,8° e 14,1° dependendo do movimento avaliado.

### O que este projeto está explorando (e por isso precisa de auto-validação)

Todos os estudos encontrados que validam velocidade via vídeo rastreiam a própria barra (por transdutor, laser ou marcador visual na barra ou máquina). Nenhum usa landmark corporal (quadril, ombro) como proxy do trajeto da barra, que é a abordagem do RepDecay, aproveitando que o MediaPipe já entrega isso de graça via "world landmarks" (coordenadas 3D em metros reais, com origem no ponto médio dos quadris).

Isso não invalida a ideia (o quadril se move de forma muito correlacionada com a barra num agachamento), mas significa que essa parte específica do projeto é exploratória, não uma técnica pronta que está sendo só implementada. Por isso o escopo inclui uma etapa de auto-validação (seção 8), inspirada no desenho de "validade concorrente" usado nos estudos reais: comparar a métrica automática com uma referência manual simples, documentar a diferença, e ser transparente sobre isso no README.

## 3. Escopo da V1

**Entra na V1:**
- Um exercício só (agachamento livre, câmera lateral)
- Processamento de vídeo gravado (não precisa ser tempo real)
- Velocidade concêntrica calculada via world landmarks (m/s), não só ângulo do joelho
- Registro manual de peso, repetições totais e se a série foi até a falha real
- Geração de rótulo real de RIR a partir de séries até a falha (não RIR percebido/subjetivo)
- Modelo de calibração pessoal por usuário e exercício, com fallback populacional (baseado na literatura) pra usuário novo
- Comparação entre modelo simples (só velocidade) e modelo com mais variáveis (profundidade, tempo de execução)
- Detecção de repetição robusta a ruído (suavização + detecção de pico por proeminência)
- Validação do modelo sem vazamento de dado (split por sessão, métrica MAE)
- Etapa de auto-validação da métrica de velocidade contra uma referência manual
- Banco de dados já estruturado pra múltiplos exercícios (populado só com agachamento)
- Dashboard de evolução do usuário

**Fica pra V2 (documentar no README como próximos passos):**
- Múltiplos exercícios (supino, terra)
- Processamento em tempo real (webcam ao vivo)
- Modelo preditivo mais sofisticado (ex: redes neurais)
- App mobile

## 4. Stack técnica (100% gratuita)

| Camada | Ferramenta | Custo | Observação |
|---|---|---|---|
| Linguagem | Python 3.11+ | Gratuito | |
| Visão computacional | MediaPipe Pose (Google) | Gratuito | Landmarks normalizados + world landmarks (3D em metros) |
| Processamento de vídeo | OpenCV | Gratuito | Leitura de frames, overlay visual pra debug |
| Cálculo/dados | NumPy + Pandas | Gratuito | Cálculo de ângulos, velocidade, manipulação de séries |
| Suavização e detecção de pico | SciPy | Gratuito | `savgol_filter`, `find_peaks` com prominence |
| Banco de dados | SQLite | Gratuito | Local, sem servidor |
| Modelo estatístico | scikit-learn | Gratuito | Regressão linear (baseline e modelo pessoal) |
| Dashboard | Streamlit ou Tableau Public | Gratuito | Tableau já é ferramenta usada no Global Retail Pulse |
| Versionamento | Git + GitHub | Gratuito | Repositório público pro portfólio |
| Ambiente | Jupyter Notebook (exploração) + scripts .py (final) | Gratuito | |

Hardware necessário: só celular ou webcam. Nenhuma ferramenta paga em nenhuma etapa.

## 5. Arquitetura do banco de dados

```sql
CREATE TABLE exercicio (
    id INTEGER PRIMARY KEY,
    nome TEXT,
    articulacao_principal TEXT,
    angulo_min_esperado REAL,
    angulo_max_esperado REAL
);

CREATE TABLE usuario (
    id INTEGER PRIMARY KEY,
    nome TEXT
);

CREATE TABLE serie (
    id INTEGER PRIMARY KEY,
    usuario_id INTEGER REFERENCES usuario(id),
    exercicio_id INTEGER REFERENCES exercicio(id),
    peso_kg REAL,
    data TEXT,
    reps_totais INTEGER,
    foi_ate_falha BOOLEAN,      -- true = série AMRAP, gera rótulo real de RIR
    tipo_sessao TEXT            -- 'treino' ou 'auto_validacao'
);

CREATE TABLE repeticao (
    id INTEGER PRIMARY KEY,
    serie_id INTEGER REFERENCES serie(id),
    numero_rep INTEGER,
    velocidade_concentrica_ms REAL,   -- via world landmarks, em m/s
    velocidade_angular_joelho REAL,   -- métrica secundária
    angulo_max REAL,
    angulo_min REAL,                  -- proxy de profundidade
    duracao_concentrica REAL,         -- em segundos
    timestamp_inicio REAL,
    timestamp_fim REAL,
    rir_real INTEGER,                 -- só preenchido quando serie.foi_ate_falha = true
    rir_estimado REAL                 -- saída do modelo, preenchido sempre
);
```

## 6. Pipeline técnico completo

1. Captura de vídeo (arquivo gravado) via OpenCV, câmera lateral (perfil)
2. MediaPipe Pose extrai landmarks normalizados e world landmarks (quadril, joelho, tornozelo, ombro) frame a frame
3. Suaviza os sinais de ângulo e posição vertical (filtro Savitzky-Golay via SciPy) pra reduzir ruído de detecção
4. Calcula o ângulo do joelho por frame (define início/fim da repetição e mede profundidade)
5. Calcula o deslocamento vertical do ponto médio entre os quadris (via world landmarks) e converte em velocidade concêntrica, em m/s
6. Detecta repetições por picos e vales com proeminência (`scipy.signal.find_peaks`) no sinal suavizado, evitando repetição fantasma
7. Por repetição, calcula: velocidade concêntrica (m/s), profundidade (ângulo mínimo), duração da fase concêntrica
8. Registro manual do usuário: peso, reps totais, se a série foi até a falha real
9. Se foi até a falha: gera `rir_real` de cada rep retroativamente (última rep = 0, penúltima = 1, e assim por diante)
10. Persiste tudo no SQLite
11. Se já existe histórico suficiente do usuário e exercício: treina/atualiza a regressão pessoal (velocidade e outras variáveis → RIR) usando as séries rotuladas por falha real
12. Se não existe histórico suficiente (usuário novo): usa como ponto de partida os valores de referência da literatura (Paulsen et al., 2025) como modelo populacional
13. Aplica o modelo (pessoal ou populacional) pra estimar `rir_estimado` nas séries que não foram até a falha
14. Dashboard mostra evolução, curva de fadiga por treino, e comparação entre modelo simples (só velocidade) e modelo com mais variáveis

## 7. Metodologia de modelagem e validação

- **Geração de rótulo real:** ao invés de depender de RIR percebido (subjetivo, é o que a própria literatura usa e reconhece como limitação), o projeto gera RIR real a partir de séries que o usuário efetivamente levou até a falha. Isso é mais raro na literatura publicada e é um ponto forte de rigor do projeto.
- **Modelo baseline:** usa os valores de referência da literatura (por exemplo, velocidade de aproximadamente 0,70 m/s no início de uma série a 79% de 1RM no agachamento, caindo pra cerca de 0,49 m/s perto de 89% de 1RM, segundo Paulsen et al., 2025) como ponto de partida populacional, sem nenhum dado do usuário ainda.
- **Modelo pessoal:** regressão treinada especificamente nos dados daquele usuário, naquele exercício, conforme o histórico de séries até a falha cresce.
- **Modelo com mais variáveis:** adiciona profundidade e duração da fase concêntrica à velocidade, comparando o desempenho preditivo com o modelo simples.
- **Validação sem vazamento:** separação treino/teste por sessão (data do treino), nunca por repetição individual, já que repetições da mesma série são correlacionadas entre si.
- **Métrica de avaliação:** erro médio absoluto (MAE) entre `rir_estimado` e `rir_real`, calculado nas séries de teste (séries até a falha que ficaram de fora do treino do modelo).

## 8. Protocolo de uso e etapa de auto-validação

**Uso recomendado:** sugerir que o usuário faça periodicamente (não toda sessão) uma série até a falha real, só pra alimentar o modelo com rótulo confiável. As demais séries usam o modelo já treinado pra estimar RIR sem precisar chegar na falha.

**Auto-validação da métrica de velocidade (fazer pelo menos uma vez, documentar no README):**
1. Gravar um pequeno conjunto de repetições (5 a 10) em vídeo
2. Nesses mesmos vídeos, marcar manualmente (visualmente, quadro a quadro) o início e fim da fase concêntrica, do jeito que os apps de celular validados na literatura fazem
3. Calcular a velocidade a partir dessa marcação manual e comparar com a velocidade automática obtida via landmark corporal (world landmarks)
4. Documentar a diferença encontrada (erro médio, por exemplo) como parte da validação do próprio método

Isso reproduz, numa escala pequena e pessoal, o mesmo tipo de desenho de "validade concorrente" usado nos estudos reais de VBT, e é o que transforma o projeto de "achei que ia funcionar" pra "testei e documentei o quanto funciona".

## 9. Limitações a documentar no README

- O sistema estima proximidade da falha pela cinemática do movimento, não mede força real nem ativação muscular
- Landmark corporal (quadril) é usado como proxy do trajeto da barra, não é o mesmo que rastrear a barra diretamente. Essa é uma adaptação exploratória, validada apenas de forma própria e em pequena escala (seção 8), não uma técnica com validação científica direta na literatura encontrada
- Pose estimation sem marcador tem erro sistemático documentado na literatura de biomecânica: cerca de 30 a 50mm na localização de quadril e joelho, e de 2,8° a 14,1° em ângulo articular, dependendo do movimento
- Mesmo em condições ideais de estudo controlado, a relação entre velocidade e RIR varia bastante entre pessoas (correlação de 0,3 a 0,9 conforme o indivíduo), então a precisão do modelo tende a melhorar com mais dado pessoal acumulado, não é fixa
- A precisão depende diretamente de quantas séries até a falha o usuário registra; sem isso, o sistema fica no modelo populacional genérico
- V1 processa vídeo gravado, não é feedback em tempo real durante o treino

## 10. Ordem de implementação sugerida

1. Script de teste: MediaPipe Pose extraindo landmarks normalizados e world landmarks num vídeo de agachamento, validação visual
2. Cálculo do ângulo do joelho + suavização do sinal (SciPy)
3. Detecção automática de repetições por proeminência de pico
4. Cálculo de velocidade concêntrica em m/s via world landmarks (quadril)
5. Etapa de auto-validação (comparar com marcação manual em poucas reps)
6. Modelagem e criação do banco SQLite completo
7. Script/formulário de registro manual (peso, reps totais, se foi até a falha)
8. Lógica de geração de `rir_real` pra séries até a falha
9. Modelo baseline (literatura) + modelo pessoal (regressão por usuário/exercício)
10. Modelo com variáveis adicionais e comparação com o modelo simples
11. Validação com split por sessão e cálculo de MAE
12. Dashboard (Streamlit ou Tableau Public)
13. Documentação final (README com metodologia, referências, limitações e resultado da auto-validação)

## 11. Por que isso é forte pro portfólio

- Resolve um problema real e documentado (acesso a VBT), não é só "achei legal"
- Metodologia fundamentada em literatura científica revisada por pares, com fontes citadas
- Distingue claramente o que é validado na ciência do que é exploração própria do projeto, e inclui uma etapa real de auto-validação pra isso, postura que qualquer recrutador técnico reconhece como maturidade
- Pipeline completo: dado não estruturado (vídeo) → dado estruturado (banco relacional) → geração de rótulo real → modelo treinado e validado → visualização
- Toda a stack é gratuita e replicável, fácil de documentar e publicar no GitHub
- Sai do varejo (diversifica em relação ao Global Retail Pulse) e mostra domínio de visão computacional, dados e leitura crítica de literatura científica

## 12. Referências

**Artigos revisados por pares:**

- Paulsen et al. (2025). Exercise type, training load, velocity loss threshold, and sets affect the relationship between lifting velocity and perceived repetitions in reserve in strength-trained individuals. *PeerJ*, 13, e19797. https://doi.org/10.7717/peerj.19797
- Concurrent validity of novel smartphone-based apps monitoring barbell velocity in powerlifting exercises. *PLOS ONE* (2024). https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0313919
- Video-Based System for Automatic Measurement of Barbell Velocity in Back Squat. *Sensors*, 21(3), 925 (2021). https://www.mdpi.com/1424-8220/21/3/925
- Kosourikhina, V., Kavanagh, D., Richardson, M. J., & Kaplan, D. M. (2022). Validation of deep learning-based markerless 3D pose estimation. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0276258
- The accuracy of several pose estimation methods for 3D joint centre localisation (2021). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8526586/
- Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics, Part 2: Accuracy. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9002957/

**Documentação técnica:**

- MediaPipe Pose Landmarker (Google AI Edge), documentação sobre world landmarks (coordenadas 3D reais em metros). https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

**Fonte prática (não revisada por pares, usada só pra contextualizar preço de hardware):**

- Velocity Based Strength Training for Rowing: Part 1. https://rowingstronger.substack.com/p/velocity-based-strength-training

Antes de citar formalmente esses artigos num README ou entrevista, vale abrir cada DOI/link e conferir a lista completa de autores, já que de alguns eu só consegui confirmar o sobrenome do primeiro autor ou nem isso.
