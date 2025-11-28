import cv2
import numpy as np
import sys
import os

# ---------------------------------------------------------
# PARÂMETROS DE DECISÃO (AJUSTE SE PRECISAR)
# ---------------------------------------------------------
# Regra "forte": muitos matches e fração razoável
MIN_GOOD_STRONG = 15       # nº mínimo de good_matches
MIN_RATIO_STRONG = 0.005   # good_matches / min_keypoints

# Regra "fraca": poucos matches, mas ainda aceitáveis
# (pensada para casos tipo as basílicas)
MIN_GOOD_WEAK = 7
MIN_RATIO_WEAK = 0.0055


def decidir_mesmo_local(kp1, kp2, good_matches):
    """Decide se são duas visões do mesmo local usando apenas os matches."""
    n1 = len(kp1)
    n2 = len(kp2)
    good = len(good_matches)
    min_kp = max(1, min(n1, n2))
    ratio = good / min_kp

    print(f"Total de keypoints img1: {n1}")
    print(f"Total de keypoints img2: {n2}")
    print(f"Boas correspondências (teste de razão): {good}")
    print(f"Razão good_matches / min_keypoints = {ratio:.5f}")

    # Regra forte
    cond_strong = (good >= MIN_GOOD_STRONG) and (ratio >= MIN_RATIO_STRONG)

    # Regra fraca (fallback)
    cond_weak = (good >= MIN_GOOD_WEAK) and (ratio >= MIN_RATIO_WEAK)

    same_place = cond_strong or cond_weak

    if same_place:
        print("Conclusão: provavelmente SÃO duas visões do MESMO local.")
    else:
        print("Conclusão: provavelmente NÃO são visões do mesmo local.")


def processar_imagens(img1_path, img2_path, out_matches_path="matches.png"):
    # 1) Ler imagens
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Erro ao carregar as imagens.")
        return

    # Converter para escala de cinza
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2) Detectar keypoints + descritores com ORB
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print("Não foi possível extrair descritores suficientes.")
        return

    # (opcional) salvar imagens com keypoints
    img1_kp = cv2.drawKeypoints(
        img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img2_kp = cv2.drawKeypoints(
        img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite("pontos_img1.png", img1_kp)
    cv2.imwrite("pontos_img2.png", img2_kp)

    # 3) Matching: BFMatcher + KNN + teste de razão de Lowe
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio_lowe = 0.75  # limiar do teste de razão
    for m, n in matches_knn:
        if m.distance < ratio_lowe * n.distance:
            good_matches.append(m)

    # 4) Desenhar matches (todos os good_matches, ou os N melhores)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    num_to_draw = min(50, len(good_matches))
    img_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches[:num_to_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(out_matches_path, img_matches)
    print(f"Imagem de correspondências salva em: {out_matches_path}")

    # 5) Decisão final (apenas com contagem de matches)
    decidir_mesmo_local(kp1, kp2, good_matches)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        prog = os.path.basename(sys.argv[0])
        print(f"Uso: python {prog} imagem1 imagem2 [saida_matches.png]")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) >= 4 else "matches.png"

    processar_imagens(img1_path, img2_path, out_path)
