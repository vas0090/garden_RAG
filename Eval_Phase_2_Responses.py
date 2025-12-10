from typing import List, Tuple, Dict, Iterable
import numpy as np
import re
import statistics

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers") from e

def simple_sentence_split(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    parts = re.split(r'[.?!\n]+', text)
    parts = [p.strip() for p in parts if p and p.strip()]   0000000000000
    return parts

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    def safe_norm(x, axis=1, keepdims=True):
        n = np.linalg.norm(x, axis=axis, keepdims=keepdims)
        n[n == 0] = 1e-12
        return n
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    A_norm = A / safe_norm(A, axis=1, keepdims=True)
    B_norm = B / safe_norm(B, axis=1, keepdims=True)
    return np.dot(A_norm, B_norm.T)

class Evaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if len(texts) == 0:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def semantic_coverage(self, gold_spans: List[str], generated_text: str) -> float:
        if not gold_spans:
            return 0.0
        gen_emb = self.embed([generated_text])            # (1, d)
        gold_embs = self.embed(gold_spans)                # (m, d)
        sims = cosine_sim_matrix(gold_embs, gen_emb).squeeze(axis=1)  # (m,)
        return float(np.mean(sims)) if sims.size > 0 else 0.0

    def bertscore_style(self, gold_text: str, generated_text: str) -> float:
        gold_sents = simple_sentence_split(gold_text)
        gen_sents = simple_sentence_split(generated_text)
        if not gold_sents or not gen_sents:
            return 0.0
        g_emb = self.embed(gold_sents)   # (m, d)
        r_emb = self.embed(gen_sents)    # (n, d)
        sims = cosine_sim_matrix(g_emb, r_emb)  # (m, n)
        best_for_gold = sims.max(axis=1) if sims.size > 0 else np.array([])
        best_for_gen  = sims.max(axis=0) if sims.size > 0 else np.array([])
        P_prime = float(best_for_gold.mean()) if best_for_gold.size > 0 else 0.0
        R_prime = float(best_for_gen.mean()) if best_for_gen.size > 0 else 0.0
        if (P_prime + R_prime) == 0.0:
            return 0.0
        f1 = 2.0 * P_prime * R_prime / (P_prime + R_prime)
        return float(f1)

    def partial_correctness(self, gold_spans: List[str], generated_text: str) -> float:
        if not gold_spans:
            return 0.0
        gen_sents = simple_sentence_split(generated_text)
        if not gen_sents:
            return self.semantic_coverage(gold_spans, generated_text)
        gold_embs = self.embed(gold_spans)   # (m, d)
        gen_embs  = self.embed(gen_sents)    # (n, d)
        sims = cosine_sim_matrix(gold_embs, gen_embs)  # (m, n)
        best_per_gold = sims.max(axis=1) if sims.size > 0 else np.zeros(len(gold_spans))
        return float(best_per_gold.mean())

DATASET: Dict[str, Dict] = {
        "Q1": {
            "query": "What regional climate and soil pH conditions produce optimal potato yields?",
            "ground_truth": ["Potatoes perform best in cool temperate climates with daytime temperatures between 15–20°C and minimal frost exposure. Optimal soil pH ranges from 5.0–6.0 to prevent scab disease. Well-drained, loamy soils with moderate moisture retention are ideal for tuber expansion and nutrient uptake."],
            "retrieved": ["The ideal soil pH for potatoes is between 6 and 6.5, and they can tolerate soil with a pH as low as 5. The best soil temperature for planting potatoes is between 10°C and 30°C. Regional climate and soil pH conditions that produce optimal potato yields include a temperate climate with cool summers and cold winters, as well as well-drained, fertile soil with a pH of 6 to 7."]
        },
        "Q2": {
            "query": "What methods are most effective for preventing insect infestations in strawberry plants without chemical pesticides?",
            "ground_truth": ["Effective non-chemical controls include using neem oil sprays weekly during early fruiting, introducing ladybugs for aphid control, and installing fine mesh netting to deter fruit flies. Intercropping with marigolds also reduces nematode and beetle pressure under humid conditions."],
            "retrieved": ["Some effective methods for preventing insect infestations in strawberry plants without chemical pesticides include: 1. Planting resistant varieties: Choose strawberry varieties that are known for their resistance to pests and diseases. 2. Good garden hygiene: Keep the garden clean and free from weeds and debris. 3. Companion planting: Plant companion plants such as marigolds, basil, and garlic around strawberry plants to repel pests. 4. Handpicking: Remove any pests you see by hand, including eggs, larvae, and adults. 5. Use organic pest control methods: Use natural predators like ladybirds and lacewings to manage pests. 6. Use organic fertilizers: Use organic fertilizers to provide the plants with the nutrients they ."]
        },
        "Q3": {
            "query": "Why do hydrangea flowers change color each year, and how can I control the shade?",
            "ground_truth": ["Hydrangea color shifts occur because anthocyanin pigments react to soil acidity. Acidic soils (pH 5.0 to 5.5) produce blue blooms due to aluminum availability, while alkaline soils (pH 6.5 to 7.0) yield pink shades. Adding garden lime raises pH for pink tones; sulfur lowers it for blue."],
            "retrieved": ["Hydrangea flowers change color each year due to changes in soil pH, sunlight exposure, and fertilizer levels. To control the shade, you can adjust the soil pH, provide partial shade, and apply a balanced fertilizer. Additionally, pruning can help promote bushier growth and increase flower production."]
        },
        "Q4": {
            "query": "How should I prepare my garden soil and perennials for winter to promote vigorous spring regrowth?",
            "ground_truth": ["Cut back dead foliage after the first frost, apply 5–8 cm of organic mulch to insulate roots, and sow cover crops like clover to prevent erosion and fix nitrogen. Avoid heavy watering before freeze events to reduce heaving and root damage."],
            "retrieved": ["To promote vigorous spring regrowth in your garden, prepare your soil and perennials for winter by adding organic matter, such as compost or well-rotted manure, to improve its structure and fertility. This will help retain moisture, suppress weeds, and provide essential nutrients for the plants. Additionally, cover the beds with a layer of mulch, such as straw or bark chips, to protect the roots from freezing and thawing. Finally, remove any dead or diseased plants from the garden to prevent the spread of pests and diseases."]
        },
        "Q5": {
            "query": "Which vegetable pairs exhibit beneficial companion-planting relationships that improve soil nutrients and pest resistance?",
            "ground_truth": ["Beans and corn form a classic symbiotic pair: beans fix nitrogen, enriching soil for corn. Basil deters tomato hornworms when planted beside tomatoes, while carrots and onions repel each other’s pests through complementary volatile compounds."],
            "retrieved": ["Beans and peas are excellent companion plants for lettuce because they fix nitrogen in the soil, which is essential for healthy lettuce growth. Additionally, beans can repel pests such as aphids and cucumber beetles, while lettuce can attract beneficial insects such as ladybugs and lacewings. Together, these plants create a symbiotic relationship that benefits both the plants and the environment."]
        },
        "Q6": {
            "query": "What visual and growth indicators show that an indoor plant is thriving and well-adjusted to its environment?",
            "ground_truth": ["Healthy indoor plants show firm, turgid leaves, vivid coloration, and steady new leaf or bud formation. Shiny leaves and consistent growth rates indicate proper humidity and sufficient light exposure. Brown tips or stunted growth suggest imbalance."],
            "retrieved": ["New growth, a healthy leaf count, and new buds are all indicators of a thriving indoor plant. Additionally, plants that receive adequate water, sunlight, and fertilizer show signs of health, such as leaf drop or increased growth. Plants with pest infestations or low humidity levels may need more attention and care. Regular monitoring and maintenance can help ensure that plants remain healthy and vibrant."]
        },
        "Q7": {
            "query": "How frequently should I rotate crops in a home vegetable garden, and what crop families should follow each other to maintain soil fertility?",
            "ground_truth": ["Rotate crops annually to prevent nutrient depletion. Follow nitrogen-fixing legumes (peas, beans) with root crops (carrots, onions), and then fruiting crops (tomatoes, peppers). Avoid repeating family groups within three years to minimize disease buildup."],
            "retrieved": ["Crop rotation should be done every 3-5 years, and crop families should follow each other to maintain soil fertility. For example, you can plant legumes (such as beans or peas) in one bed and then follow them with a bed of brassicas (such as broccoli or cabbage) to improve soil structure and reduce disease. This cycle can be repeated for multiple growing seasons, providing a balanced and healthy garden."]
        },
        "Q8": {
            "query": "What are the most common causes of leaf yellowing and wilting in indoor plants, and how can they be corrected?",
            "ground_truth": ["Yellowing leaves usually result from overwatering or nitrogen deficiency. Allow the soil to dry between waterings and apply a balanced fertilizer (10-10-10). Wilting under dry conditions signals water stress; increase humidity or watering frequency accordingly."],
            "retrieved": ["The most common causes of leaf yellowing and wilting in indoor plants are overwatering, underwatering, and nutrient deficiencies. To correct these issues, it's important to water plants when the soil is dry to the touch, provide them with the appropriate amount of water, and fertilize them with a balanced fertilizer. Additionally, it's important to provide plants with proper sunlight, humidity, and temperature conditions. If you notice any signs of yellowing or wilting, it's best to take immediate action to correct the issue before it becomes more severe."]
        },
        "Q9": {
            "query": "Can a plant survive or grow in complete darkness, and what physiological processes are affected?",
            "ground_truth": ["Plants cannot grow indefinitely in total darkness because photosynthesis ceases without light. They exhibit etiolation, elongated stems and pale leaves, caused by chlorophyll loss and energy depletion. Prolonged darkness leads to carbohydrate starvation and death."],
            "retrieved": ["Plants can survive in complete darkness for a short period, but their physiological processes are significantly affected. Without light, plants cannot perform photosynthesis, which is essential for producing energy and nutrients. They also cannot regulate their temperature, water balance, or growth hormones. Additionally, plants may experience stress and damage to their cells and tissues."]
        },
        "Q10": {
            "query": "How should I design a raised bed garden to optimize drainage, root health, and overall yield?",
            "ground_truth": ["A well-designed raised bed should be 25–35 cm deep with layered soil: coarse gravel at the base for drainage, followed by loam and compost for aeration and nutrients. A slight 2–3% slope promotes runoff while preventing root rot and maintaining high yields."],
            "retrieved": ["To optimize drainage, root health, and overall yield in a raised bed garden, it's important to choose a location with good drainage, incorporate organic matter to improve soil structure and fertility, and plant crops that are suited to the growing conditions. Additionally, consider using raised beds with a depth of at least 12 inches to provide ample space for roots to grow. Finally, regularly check the soil moisture and adjust watering as needed to prevent overwatering or underwatering."]
        },
    }
def main():
    evaluator = Evaluator()
    rows = []

    for qid in sorted(DATASET.keys(), key=lambda x: int(x[1:])):  # Q1..Q10 order
        item = DATASET[qid]
        gold_spans = item.get("ground_truth", [])
        retrieved_list = item.get("retrieved", [])
        generated_text = " ".join([r.strip() for r in retrieved_list if r and r.strip()])

        coverage = evaluator.semantic_coverage(gold_spans, generated_text)
        bert_f1  = evaluator.bertscore_style(" ".join(gold_spans), generated_text)
        partial  = evaluator.partial_correctness(gold_spans, generated_text)
        rows.append((qid, coverage, bert_f1, partial))

    print(f"{'QueryID':<6} | {'Coverage':>8} | {'BERT-F1':>8} | {'Partial':>8}")
    print("-" * 52)
    for qid, cov, bert, part in rows:
        print(f"{qid:<6} | {cov:8.3f} | {bert:8.3f} | {part:8.3f}")

    avg_cov = statistics.mean(r[1] for r in rows)
    avg_bert = statistics.mean(r[2] for r in rows)
    avg_part = statistics.mean(r[3] for r in rows)
    print("-" * 52)
    print(f"{'AVERAGE':<6} | {avg_cov:8.3f} | {avg_bert:8.3f} | {avg_part:8.3f}")

if __name__ == "__main__":
    main()