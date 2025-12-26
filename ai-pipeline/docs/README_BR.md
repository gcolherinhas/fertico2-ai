# FertiCO₂-AI

FertiCO₂-AI é um framework computacional aberto, modular e totalmente
reprodutível, desenvolvido para classificar, predizer e interpretar o potencial
de absorção de CO₂ em líquidos iônicos de aminoácidos (AAILs) utilizando
simulações de dinâmica molecular e inteligência artificial.

O projeto integra descritores físico-químicos obtidos via GROMACS com modelos
de aprendizado de máquina capazes de aprender regimes ordinais de absorção
e quantificar incertezas preditivas.

---

##  Escopo do Projeto

Este repositório disponibiliza:

- Um pipeline completo de simulações de dinâmica molecular (MD) para sistemas AAIL e 
AAIL–CO₂
- Extração automatizada de descritores estruturais, energéticos, dinâmicos e de ligações de 
hidrogênio
- Classificação não supervisionada fisicamente ordenada de regimes de absorção de CO₂
- Modelos supervisionados de IA ordinal com análise explícita de incerteza
- Total reprodutibilidade, transparência e aderência à ciência aberta

---

##  Documentação Técnica Principal

A descrição completa e detalhada do pipeline de Dinâmica Molecular,
incluindo preparação dos sistemas, protocolos de simulação, extração de
descritores e interpretação física, está disponível no documento abaixo:

Documentação Completa do Pipeline de MD (Português)  
`FERTICO2AI-AI_PIPLINE_BR.docx`

Este documento deve ser considerado a referência técnica principal para:
- Reprodução das simulações
- Compreensão física dos descritores
- Interpretação das predições da IA
- Extensão ou adaptação do workflow

---

## Pipeline de Inteligência Artificial

O pipeline de IA (Estágios 1–3), incluindo clustering, treinamento da rede neural,
análise ordinal e quantificação de incerteza, está implementado diretamente nos
scripts do repositório e documentado no próprio código.

---

## Reprodutibilidade e Ciência Aberta

- Todos os scripts são integralmente reprodutíveis
- Modelos, escaladores e mapeamentos físicos são explicitamente salvos
- Os descritores possuem interpretação físico-química clara
- O projeto segue princípios de ciência aberta, IA responsável e
  transparência metodológica

---

## Licença

Este projeto é distribuído sob a Licença MIT.  
Consulte o arquivo `LICENSE` para mais informações.

