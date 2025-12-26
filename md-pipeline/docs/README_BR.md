# FertiCO₂-AI — Molecular Dynamics Pipeline

---

## Versão em Português

### FertiCO₂-AI — Pipeline de Dinâmica Molecular (MD)

O FertiCO₂-AI é um framework computacional aberto, modular e totalmente reprodutível 
para o estudo e a predição do potencial de absorção de CO₂ por líquidos iônicos de 
aminoácidos (AAILs).

Este repositório concentra exclusivamente o pipeline de Dinâmica Molecular (MD) utilizado 
para gerar os dados físico-químicos que alimentam os modelos de Inteligência Artificial do 
projeto.

---

## Objetivo do Pipeline de MD

O pipeline de Dinâmica Molecular tem como objetivos principais:

- Simular AAILs puros e sistemas AAIL–CO₂
- Obter descritores físico-químicos, estruturais, dinâmicos e energéticos
- Garantir consistência física, reprodutibilidade e rastreabilidade
- Fornecer dados confiáveis para:
  - Classificação física não supervisionada
  - Treinamento de modelos de IA supervisionados

---

## Documentação Técnica Oficial

Este repositório é acompanhado por um documento técnico completo, que DEVE ser 
consultado antes da execução do pipeline:

### Documento principal

- `docs/FERTICO2AI-MD_PIPELINE_BR.pdf`

Este documento descreve detalhadamente:

- Construção das configurações iniciais com Packmol
- Definição de topologia e campo de força (CHARMM36)
- Protocolos de minimização e equilibração
- Inserção automatizada de moléculas de CO₂
- Simulações de produção
- Extração de descritores com comandos do GROMACS
- Boas práticas de reprodutibilidade

Este README funciona como um guia operacional.  
O documento técnico é a referência metodológica completa.

---

## Etapas do Pipeline de Dinâmica Molecular

### 1 Construção do Sistema Inicial

- Construção do AAIL puro ou de misturas
- Neutralidade elétrica garantida
- Quantidade típica:
  - 300 pares iônicos (AAIL puro)
  - Misturas variando de 50–250 a 250–50 pares de 50 em 50 para cada par da mistura

---

### 2 Topologia e Campo de Força

- Campo de força: CHARMM36
- Arquivo principal: `topol.top`
- Definição explícita de:
  - Tipos atômicos
  - Cargas parciais
  - Interações ligadas e não ligadas

---

### 3 Minimização e Equilibração do AAIL Puro

- Minimização de energia
- Equilibração em NVT
- Equilibração em NPT
- Monitoramento de:
  - Energia
  - Temperatura
  - Pressão
  - Densidade

---

### 4 Inserção Automatizada de CO₂

Após a equilibração do AAIL puro:

- Expansão da caixa no eixo Z
- Inserção explícita de 300 moléculas de CO₂
- Preservação da configuração original do líquido iônico

---

### 5 Simulações de Produção AAIL–CO₂

- Simulações longas em NPT ou NVT
- Geração de trajetórias (`.xtc` ou `.trr`)
- Geração de arquivos de energia (`.edr`)

---

### 6 Extração de Descritores

Os principais descritores extraídos incluem:

- Energias Coulomb e Lennard-Jones (AAIL–CO₂)
- Densidade
- Coeficiente de difusão (MSD)
- Ligações de hidrogênio:
  - Número
  - Tempo de vida
  - Energia efetiva de dissociação

---

##  Licença

Este projeto está lançado sobre a MIT License.

---

##  Citação

Se você user estes arquivos para trabalhos acadêmicos, por favor, 
Entre em contato com os administradores do FertiCO2-AI

---


