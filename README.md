# PILT-FLASK
# PROJETO PILT

## Integração Flask + Pipeline de Processamento de Imagens

**Autores:** Guilherme Schnirmann e Sâmela Soraia Sartin

---

## Sumário

- [1. Visão Geral](#1-visão-geral)
- [2. Estrutura Esperada do Projeto](#2-estrutura-esperada-do-projeto)
- [3. Pré-requisitos no Windows](#3-pré-requisitos-no-windows)
- [4. Como Clonar o Projeto do GitHub](#4-como-clonar-o-projeto-do-github)
- [5. Como Criar e Ativar o Ambiente Virtual (venv)](#5-como-criar-e-ativar-o-ambiente-virtual-venv)
- [6. Atualizar o pip](#6-atualizar-o-pip)
- [7. Instalar as Dependências](#7-instalar-as-dependências)
- [8. Arquivo `requirements.txt`](#8-arquivo-requirementstxt)
- [9. Modelos Obrigatórios](#9-modelos-obrigatórios)
- [10. Como Rodar a Aplicação](#10-como-rodar-a-aplicação)
- [11. Como Usar a Interface](#11-como-usar-a-interface)
- [12. Onde os Resultados São Salvos](#12-onde-os-resultados-são-salvos)
- [13. Endpoints da Aplicação](#13-endpoints-da-aplicação)
- [14. Teste via Navegador](#14-teste-via-navegador)
- [15. Teste via Terminal (Opcional)](#15-teste-via-terminal-opcional)
- [16. Problemas Comuns](#16-problemas-comuns)
- [17. Boas Práticas](#17-boas-práticas)
- [18. Arquivo `.gitignore` Sugerido](#18-arquivo-gitignore-sugerido)
- [19. Como Encerrar a Aplicação](#19-como-encerrar-a-aplicação)
- [20. Passo a Passo Rápido](#20-passo-a-passo-rápido)
- [21. Observação Final](#21-observação-final)

---

## 1. Visão Geral

Este projeto executa uma pipeline de processamento de imagens com Flask.

O sistema permite:

- subir uma imagem por interface web
- processar a imagem com a `pipeline_core.py`
- calcular medidas do enduramento
- salvar imagens intermediárias e finais
- exibir os resultados visuais na interface

O backend foi desenvolvido em Python com Flask.  
A pipeline utiliza OpenCV, Ultralytics/YOLO, NumPy, SciPy e Matplotlib.

---

## 2. Estrutura Esperada do Projeto

A estrutura recomendada é:

```text
Projeto-PILT-Flask/
│
├── app.py
├── pipeline_core.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── a4_best.pt
│   └── ppd_best.pt
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
├── uploads/
└── outputs/
```

### Importante

- Os arquivos `a4_best.pt` e `ppd_best.pt` devem estar obrigatoriamente na pasta `models`.
- As pastas `uploads` e `outputs` podem estar vazias.
- A pasta `outputs` será preenchida automaticamente durante o uso.

---

## 3. Pré-requisitos no Windows

Antes de começar, é necessário ter instalado:

1. Python 3.10 ou superior  
2. Git (opcional, se for clonar do GitHub)  
3. VS Code (opcional, mas recomendado)

### Sugestão

- Instale o Python pelo site oficial.
- Durante a instalação, marque a opção **Add Python to PATH**.

Para verificar se o Python foi instalado corretamente, abra o Prompt de Comando (`cmd`) ou o PowerShell e execute:

```bash
python --version
```

ou

```bash
py --version
```

---

## 4. Como Clonar o Projeto do GitHub

Se o projeto estiver no GitHub, abra o terminal e execute:

```bash
git clone URL_DO_REPOSITORIO
```

Exemplo:

```bash
git clone https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git
```

Depois, entre na pasta do projeto:

```bash
cd NOME-DO-REPOSITORIO
```

---

## 5. Como Criar e Ativar o Ambiente Virtual (venv)

Na raiz do projeto, execute:

```bash
python -m venv venv
```

Se necessário, também pode usar:

```bash
py -m venv venv
```

Isso criará uma pasta chamada `venv` com o ambiente virtual.

### Ativação no Windows - CMD

```bash
venv\Scripts\activate
```

### Ativação no Windows - PowerShell

```powershell
venv\Scripts\Activate.ps1
```

Se no PowerShell aparecer erro de permissão, execute antes:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Depois:

```powershell
venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, o terminal mostrará algo como:

```text
(venv) C:\caminho\do\projeto>
```

---

## 6. Atualizar o pip

Com o `venv` ativado, execute:

```bash
python -m pip install --upgrade pip
```

---

## 7. Instalar as Dependências

Se já existir o `requirements.txt`, execute:

```bash
pip install -r requirements.txt
```

Isso instalará todas as bibliotecas necessárias.

Se por algum motivo o `requirements.txt` não estiver disponível, instale manualmente:

```bash
pip install flask opencv-python ultralytics numpy scipy matplotlib werkzeug
```

---

## 8. Arquivo `requirements.txt`

O arquivo `requirements.txt` deve estar na raiz do projeto.

Caso seja necessário recriá-lo futuramente, com o ambiente já configurado, use:

```bash
pip freeze > requirements.txt
```

---

## 9. Modelos Obrigatórios

Antes de rodar a aplicação, verifique se estes arquivos existem:

```text
models\a4_best.pt
models\ppd_best.pt
```

Se esses arquivos não estiverem na pasta `models`, a aplicação não funcionará.

---

## 10. Como Rodar a Aplicação

Com o ambiente virtual ativado, execute:

```bash
python app.py
```

ou

```bash
py app.py
```

Se tudo estiver correto, o terminal mostrará que o Flask está rodando.

Abra o navegador e acesse:

```text
http://127.0.0.1:5000
```

---

## 11. Como Usar a Interface

Na interface web:

1. Clique no botão de seleção de arquivo  
2. Escolha uma imagem do computador  
3. Confira a pré-visualização da imagem  
4. Clique em **Processar imagem**  
5. Aguarde o processamento  
6. Visualize:
   - área em pixels
   - área em mm²
   - raio equivalente
   - imagens de resultado
   - resposta JSON

---

## 12. Onde os Resultados São Salvos

Os arquivos gerados são salvos automaticamente na pasta:

```text
outputs/
```

Cada execução cria uma subpasta própria com um identificador único.

Exemplo:

```text
outputs/8f4c8d6e-xxxx-xxxx-xxxx/
    final_overlay_full.png
    final_mask_full.png
    roi_final_mask_roi.png
    ...
```

A pasta `uploads/` também recebe o arquivo enviado.

---

## 13. Endpoints da Aplicação

A aplicação possui os seguintes endpoints:

### Página principal

```http
GET /
```

Abre a interface web.

### Health check

```http
GET /health
```

Retorna:

```json
{"status":"running"}
```

### Processamento

```http
POST /process-image
```

Recebe a imagem enviada pelo formulário.

### Acesso aos arquivos gerados

```http
GET /outputs/<request_id>/<filename>
```

---

## 14. Teste via Navegador

A forma mais simples de usar é pelo navegador:

```text
http://127.0.0.1:5000
```

---

## 15. Teste via Terminal (Opcional)

Se quiser testar via terminal em vez da interface, é possível usar uma ferramenta de requisição HTTP.

Exemplo conceitual:

- enviar uma imagem para `/process-image`
- receber um JSON com medidas e links dos arquivos

No Windows, isso também pode ser feito com Postman, que costuma ser mais simples para quem não quer usar linha de comando.

---

## 16. Problemas Comuns

### Erro: Python não reconhecido

Mensagem típica:

```text
'python' não é reconhecido como um comando interno...
```

**Solução:**

- reinstalar o Python
- marcar **Add Python to PATH**
- fechar e abrir o terminal novamente

### Erro: ambiente virtual não ativa

No PowerShell, pode ser bloqueio de execução de scripts.

Execute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Depois:

```powershell
venv\Scripts\Activate.ps1
```

### Erro: módulo não encontrado

Mensagem típica:

```text
ModuleNotFoundError: No module named ...
```

**Solução:**

- verificar se o `venv` está ativo
- executar:

```bash
pip install -r requirements.txt
```

### Erro: modelos não encontrados

Mensagem típica:

```text
FileNotFoundError: ... a4_best.pt
```

ou

```text
FileNotFoundError: ... ppd_best.pt
```

**Solução:**

- conferir se os arquivos `.pt` estão dentro da pasta `models`

### Erro: porta 5000 em uso

Se outra aplicação estiver usando a porta, feche o processo anterior ou mude a porta no `app.py`.

Exemplo:

```python
app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
```

Depois acesse:

```text
http://127.0.0.1:5001
```

### Erro: interface abre, mas não mostra resultados

**Possíveis causas:**

- a pipeline não gerou arquivos
- os nomes dos arquivos gerados não coincidem com os nomes esperados no `app.py`
- a imagem enviada não produziu detecção útil

**Verifique:**

- a resposta JSON
- a pasta `outputs`
- os logs do terminal

---

## 17. Boas Práticas

- Não enviar a pasta `venv` para o GitHub
- Não enviar a pasta `outputs` para o GitHub
- Não enviar a pasta `uploads` para o GitHub
- Conferir se os modelos `.pt` devem ou não ser versionados
- Usar sempre o `requirements.txt` para replicar o ambiente
- Manter a mesma estrutura de pastas

---

## 18. Arquivo `.gitignore` Sugerido

Crie um arquivo chamado `.gitignore` com o conteúdo abaixo:

```gitignore
venv/
__pycache__/
uploads/
outputs/
*.pyc
.DS_Store
```

Se **não** quiser subir os modelos:

```gitignore
models/*.pt
```

> **Atenção:**  
> Se os modelos forem necessários para outra pessoa rodar o projeto e não estiverem em outro local seguro, então não ignore os arquivos `.pt`.

---

## 19. Como Encerrar a Aplicação

No terminal onde o Flask estiver rodando, pressione:

```text
CTRL + C
```

---

## 20. Passo a Passo Rápido

1. Abrir a pasta do projeto  
2. Criar o `venv`:

   ```bash
   python -m venv venv
   ```

3. Ativar:

   ```bash
   venv\Scripts\activate
   ```

4. Atualizar pip:

   ```bash
   python -m pip install --upgrade pip
   ```

5. Instalar dependências:

   ```bash
   pip install -r requirements.txt
   ```

6. Conferir os modelos em `models/`  
7. Rodar:

   ```bash
   python app.py
   ```

8. Abrir:

   ```text
   http://127.0.0.1:5000
   ```

---

## 21. Observação Final

Se houver erro de ambiente, valide nesta ordem:

1. Python instalado  
2. `venv` ativo  
3. `requirements` instalados  
4. modelos presentes  
5. estrutura de pastas correta

---

**Fim do documento.**
