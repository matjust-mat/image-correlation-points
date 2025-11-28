# Detecção de pontos de interesse e comparação de imagens

Este projeto lê **duas imagens**, detecta **pontos de interesse** em cada uma, encontra **correspondências** entre esses pontos e traça **linhas** ligando os pontos equivalentes.  
Com base na quantidade de correspondências, o programa tenta decidir se as duas imagens são **duas visões do mesmo local** ou não.

---

## 1. O que o código faz

Dado um par de imagens (`imagem1` e `imagem2`), o programa:

1. Lê as duas imagens do disco.
2. Converte as imagens para escala de cinza.
3. Usa o algoritmo **ORB** (do OpenCV) para:
   - detectar **keypoints** (pontos de interesse),
   - extrair **descritores** desses pontos.
4. Usa o **BFMatcher** com KNN e **teste de razão de Lowe** para encontrar **correspondências (matches)** entre os descritores das duas imagens.
5. Desenha:
   - `pontos_img1.png`: imagem 1 com os keypoints desenhados;
   - `pontos_img2.png`: imagem 2 com os keypoints desenhados;
   - `matches.png` (ou outro nome indicado): duas imagens lado a lado com linhas ligando os pontos correspondentes.
6. Conta quantos *good matches* foram encontrados e calcula:

   \[
   \text{ratio} = \frac{\text{número de good matches}}{\min(\text{#keypoints img1}, \text{#keypoints img2})}
   \]

7. Usa uma heurística simples para decidir:

   - Se há **bastante correspondência**, o programa imprime:

     > Conclusão: provavelmente SÃO duas visões do MESMO local.

   - Caso contrário:

     > Conclusão: provavelmente NÃO são visões do mesmo local.

Os limiares usados na decisão são:

```python
# Regra "forte": muitos matches e fração razoável
MIN_GOOD_STRONG = 15
MIN_RATIO_STRONG = 0.005

# Regra "fraca": poucos matches, mas ainda aceitáveis
MIN_GOOD_WEAK = 7
MIN_RATIO_WEAK = 0.0055
```

Você pode ajustar esses valores conforme o conjunto de imagens que estiver usando.

## 2. Pré-requisitos

Python 3 instalado (3.8+ recomendado).

Biblioteca OpenCV para Python.

Para evitar problemas com libGL.so.1 (comum em servidores/containers), o ideal é instalar a versão headless do OpenCV.

Instalação das dependências

Em um ambiente virtual ou diretamente no sistema:

```bash
pip install opencv-python-headless numpy
```

Se quiser usar a versão completa com suporte a janelas (imshow etc.):

```bash
pip install opencv-python numpy
```

Observação: em algumas distribuições Linux, a versão completa pode exigir bibliotecas do sistema, como `libgl1` e `libglib2.0-0`.

## 3. Estrutura básica

Você deve ter pelo menos:

```
seu_projeto/
├─ app.py   # arquivo com o código fornecido
├─ img1.jpg              # primeira imagem
└─ img2.jpg              # segunda imagem
```

## 4. Como executar

No terminal, dentro da pasta do projeto:

```bash
python app.py img1.jpg img2.jpg
```

Opcionalmente, você pode escolher o nome do arquivo de saída com os matches:

```bash
python app.py img1.jpg img2.jpg saida_matches.png
```

Saídas geradas

Após rodar o programa, os seguintes arquivos serão criados no diretório atual:

- `pontos_img1.png`
    - Imagem 1 com os pontos de interesse (keypoints) destacados.

- `pontos_img2.png`
    - Imagem 2 com os pontos de interesse destacados.
- `matches.png` ou `saida_matches.png`
    - As duas imagens lado a lado, com linhas ligando os pontos correspondentes (somente os melhores matches).

Além disso, o terminal mostrará algo como:

```bash
Total de keypoints img1: 1234
Total de keypoints img2: 1100
Boas correspondências (teste de razão): 45
Razão good_matches / min_keypoints = 0.03650
Conclusão: provavelmente SÃO duas visões do MESMO local.
```

## 5. Ajustando a sensibilidade

Se o programa:

estiver dizendo que imagens do mesmo local NÃO são o mesmo local, deixe a heurística mais permissiva:

diminua `MIN_GOOD_STRONG` e `MIN_GOOD_WEAK`;

diminua `MIN_RATIO_STRONG` e `MIN_RATIO_WEAK`.

estiver aceitando muitas imagens de lugares diferentes como se fossem o mesmo, deixe mais rigoroso:

aumente esses limiares.

Edite no topo do arquivo:

```
MIN_GOOD_STRONG = ...
MIN_RATIO_STRONG = ...
MIN_GOOD_WEAK   = ...
MIN_RATIO_WEAK  = ...
```

e rode os testes novamente até chegar em um equilíbrio razoável para o seu trabalho.

## 6. Observações

O método funciona melhor com fotos reais do mesmo local, com mudanças moderadas de ângulo, escala e iluminação.

Desenhos, renders 3D ou versões muito estilizadas (aquarela, cartoon) podem ter poucos matches mesmo sendo “o mesmo objeto” para um ser humano.

Nada garante acerto de 100%; é uma heurística simples baseada em feature matching local (ORB).