# FertiCO₂-AI

**FertiCO₂-AI** é uma plataforma computacional **aberta, modular e totalmente reprodutível**
desenvolvida para **classificar e predizer o potencial de absorção de CO₂ por líquidos
iônicos de aminoácidos (AAILs)**, utilizando exclusivamente **simulações de dinâmica
molecular (MD)** e **inteligência artificial (IA)**.

O objetivo central do projeto é o desenvolvimento de um **modelo de IA fisicamente
fundamentado**, capaz de aprender relações estrutura–propriedade a partir de simulações
moleculares de primeiros princípios, viabilizando o **screening computacional e o
design orientado por dados de novos sistemas AAIL** voltados à captura de CO₂ e ao
desenvolvimento de fertilizantes sustentáveis.

---

## Escopo Científico e Metodologia

O FertiCO₂-AI integra três estágios computacionais fortemente acoplados:

### 1. Simulações de Dinâmica Molecular (MD)
As simulações de dinâmica molecular clássica são realizadas com o software **GROMACS
(código aberto)**, gerando trajetórias fisicamente consistentes de sistemas AAIL puros
e mistos, com e sem CO₂ explícito. A partir dessas simulações são extraídas propriedades
estruturais, energéticas, dinâmicas e termodinâmicas de forma **totalmente reprodutível**.

### 2. Classificação Não Supervisionada com Ordenação Física
Um método de **clusterização K-Means** é aplicado a descritores diretos de interação
AAIL–CO₂ (por exemplo, energias de Coulomb e Lennard-Jones, estatísticas de ligações de
hidrogênio).  
Os clusters obtidos são **reordenados de forma determinística com base em critérios
físicos**, especificamente a afinidade média AAIL–CO₂, resultando em **classes ordinais**
que representam níveis crescentes de absorção de CO₂, sem a imposição de rótulos
arbitrários.

### 3. Aprendizado Supervisionado e Predição com Incerteza
Uma **Rede Neural Multicamadas (MLP)** é treinada utilizando exclusivamente
**descritores físico-químicos intrínsecos dos AAILs**, como composição elementar,
densidade, difusão, interações IL–IL e propriedades de ligações de hidrogênio.  
O modelo prediz a **classe ordinal de absorção de CO₂**, incorporando técnicas de
balanceamento de classes e fornecendo **saídas probabilísticas**, que permitem a
quantificação de incertezas e a avaliação da confiabilidade física das predições.

---

## Conteúdo do Repositório

Este repositório inclui:

- **Pipelines de simulação MD** baseados no GROMACS
- **Fluxos de extração de descritores** estruturais, energéticos e dinâmicos
- **Scripts de clusterização com ordenação física** (Estágio 1)
- **Modelos de IA e pipelines de treinamento** (Estágio 2)
- **Ferramentas de interpretação física e análise de incerteza** (Estágio 3)
- **Scripts totalmente reprodutíveis** para pré-processamento, validação e inferência
- **Documentação e protocolos técnicos** que permitem a regeneração completa dos dados

Todos os componentes utilizam exclusivamente **ferramentas open-source** e bibliotecas
científicas consolidadas.

---

## Reprodutibilidade e Ciência Aberta

O FertiCO₂-AI foi concebido desde sua origem para atender aos princípios de **ciência
aberta e reprodutibilidade computacional**:

- Todo o código é desenvolvido em **Python** com bibliotecas open-source
- As simulações podem ser integralmente reproduzidas com o **GROMACS (licença GNU GPL)**
- Modelos treinados, escaladores e mapeamentos físicos são explicitamente salvos
- Não há dependência de softwares proprietários ou bases de dados fechadas
- Todo o pipeline pode ser auditado, replicado ou estendido de forma independente

O projeto está alinhado aos princípios **FAIR** (Findable, Accessible, Interoperable,
Reusable) e às melhores práticas internacionais de **IA responsável e transparente**.

---

## Considerações Éticas e Legais

Este projeto **não realiza qualquer tratamento de dados pessoais, sensíveis ou
identificáveis**. Toda a base de dados é composta exclusivamente por **descritores
moleculares e resultados de simulações computacionais**, assegurando conformidade plena
com a Lei Geral de Proteção de Dados (LGPD) e regulamentos correlatos.

---

## Autores e Colaboradores

O FertiCO₂-AI é desenvolvido de forma colaborativa por pesquisadores vinculados à
**Universidade Federal de Goiás (UFG)**:

- Prof. Dr. Guilherme Colherinhas de Oliveira  
- Prof. Dr. Wesley Bueno Cardoso  
- Prof. Dr. Tertius Lima da Fonseca  
- Prof. Dr. Leonardo Bruno Assis Oliveira  
- Ma. Karinna Mendanha Soares  
- Me. Lucas de Sousa Silva  
- Me. Henrique de Araujo Chagas  

Todos os integrantes possuem **participação intelectual equivalente** no projeto.

---

## Licença

Este projeto é distribuído sob a **Licença MIT**.

Copyright (c) 2025  
**Contribuidores do Projeto FertiCO₂-AI**  
Universidade Federal de Goiás (UFG)

Consulte o arquivo `LICENSE` para os termos completos.
