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
    parts = [p.strip() for p in parts if p and p.strip()]
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
            "retrieved": ["The ideal soil pH for potatoes is between 6 and 6.5, but they can tolerate soil with a pH as low as 5. The best soil temperature for planting potatoes is between 10°C and 30°C. Light soil is the best for growing large, smooth potatoes, but it must not get too dry to ensure good tuber growth."]
        },
        "Q2": {
            "query": "What methods are most effective for preventing insect infestations in strawberry plants without chemical pesticides?",
            "ground_truth": ["Effective non-chemical controls include using neem oil sprays weekly during early fruiting, introducing ladybugs for aphid control, and installing fine mesh netting to deter fruit flies. Intercropping with marigolds also reduces nematode and beetle pressure under humid conditions."],
            "retrieved": ["The most effective methods for preventing insect infestations in strawberry plants without chemical pesticides include: 1. Planting resistant varieties: Select strawberry varieties that are known to be resistant to common pests and diseases. 2. Handpicking pests: Regularly check your strawberry plants for pests and remove them by hand. 3. Cover crops: Plant cover crops such as clover or rye around your strawberry plants to provide natural pest control. 4. Companion planting: Plant companion plants such as marigolds, basil, or garlic around your strawberry plants to attract beneficial insects that prey on pests. 5. Integrated pest management (IPM): Use a combination of methods to control pests, such as using natural predators, pheromone traps, and biological control agents. 6. Proper irrigation: Ensure that your strawberry plants are properly irrigated to prevent water stress, which can make plants more susceptible to pests and diseases. 7. Proper fertilization: Use a balanced fertilizer to provide your strawberry plants with the nutrients they need to grow and resist pests and diseases. 8. Regular pruning: Prune your strawberry plants regularly to remove dead or diseased branches and promote healthy growth. 9. Clean up: Remove any fallen leaves or debris from your strawberry plants to prevent pests from hiding and breeding. 10. Monitor your plants: Monitor your strawberry plants regularly for any signs of pest or disease infestation and take action immediately if you notice any problems."]
        },
        "Q3": {
            "query": "Why do hydrangea flowers change color each year, and how can I control the shade?",
            "ground_truth": ["Hydrangea color shifts occur because anthocyanin pigments react to soil acidity. Acidic soils (pH 5.0 to 5.5) produce blue blooms due to aluminum availability, while alkaline soils (pH 6.5 to 7.0) yield pink shades. Adding garden lime raises pH for pink tones; sulfur lowers it for blue."],
            "retrieved": ["Hydrangeas change colors due to the acidity or alkalinity of the soil. To control the shade, you can add aluminum sulfate to the soil."]
        },
        "Q4": {
            "query": "How should I prepare my garden soil and perennials for winter to promote vigorous spring regrowth?",
            "ground_truth": ["Cut back dead foliage after the first frost, apply 5–8 cm of organic mulch to insulate roots, and sow cover crops like clover to prevent erosion and fix nitrogen. Avoid heavy watering before freeze events to reduce heaving and root damage."],
            "retrieved": ["Apply compost or organic matter to the beds and dress the top with plant food to ensure your garden thrives. This sets the stage for planting."]
        },
        "Q5": {
            "query": "Which vegetable pairs exhibit beneficial companion-planting relationships that improve soil nutrients and pest resistance?",
            "ground_truth": ["Beans and corn form a classic symbiotic pair: beans fix nitrogen, enriching soil for corn. Basil deters tomato hornworms when planted beside tomatoes, while carrots and onions repel each other’s pests through complementary volatile compounds."],
            "retrieved": ["Lettuce pairs well with carrots, onions, strawberries, beets, brassicas, radish, marigold, borage, chervil, Florence fennel, and leeks."]
        },
        "Q6": {
            "query": "What visual and growth indicators show that an indoor plant is thriving and well-adjusted to its environment?",
            "ground_truth": ["Healthy indoor plants show firm, turgid leaves, vivid coloration, and steady new leaf or bud formation. Shiny leaves and consistent growth rates indicate proper humidity and sufficient light exposure. Brown tips or stunted growth suggest imbalance."],
            "retrieved": ["New growth, dark green color, and healthy leaves."]
        },
        "Q7": {
            "query": "How frequently should I rotate crops in a home vegetable garden, and what crop families should follow each other to maintain soil fertility?",
            "ground_truth": ["Rotate crops annually to prevent nutrient depletion. Follow nitrogen-fixing legumes (peas, beans) with root crops (carrots, onions), and then fruiting crops (tomatoes, peppers). Avoid repeating family groups within three years to minimize disease buildup."],
            "retrieved": ["It is recommended to rotate crops every three years to maintain soil fertility and prevent nutrient depletion. This means that you should grow a different type of vegetable in the same plot of land every three years. Some good examples of crop families that can be rotated include legumes (such as beans, peas, and lentils), brassicas (such as broccoli, cauliflower, and cabbage), and cucurbits (such as squash, zucchini, and cucumbers). It is important to note that the specific crop families you choose to rotate should be based on your local climate and soil conditions. For example, if you live in a hot and humid climate, you may want to avoid growing legumes, as they can be susceptible to diseases. Similarly, if you have clay soil, you may want to avoid growing cucurbits, as they can be prone to root rot. In addition to crop rotation, it is also important to practice good soil management techniques, such as adding organic matter to the soil, controlling weeds, and watering the plants regularly. These techniques can help improve soil structure, retain moisture, and reduce the risk of soil-borne diseases. It is also worth mentioning that crop rotation can be a challenging practice to implement in a home vegetable garden, especially if you have a small plot of land. However, there are many resources available online and in gardening books that can help you get started."]
        },
        "Q8": {
            "query": "What are the most common causes of leaf yellowing and wilting in indoor plants, and how can they be corrected?",
            "ground_truth": ["Yellowing leaves usually result from overwatering or nitrogen deficiency. Allow the soil to dry between waterings and apply a balanced fertilizer (10-10-10). Wilting under dry conditions signals water stress; increase humidity or watering frequency accordingly."],
            "retrieved": ["The most common causes of leaf yellowing and wilting in indoor plants are overwatering and underwatering. Overwatering can cause the roots to rot, while underwatering can cause the leaves to turn yellow and wilt. To correct these issues, it is important to water the plants regularly, allowing the soil to dry out slightly between waterings. Additionally, it is important to use a well-draining soil mix and to provide the plants with adequate sunlight and humidity."]
        },
        "Q9": {
            "query": "Can a plant survive or grow in complete darkness, and what physiological processes are affected?",
            "ground_truth": ["Plants cannot grow indefinitely in total darkness because photosynthesis ceases without light. They exhibit etiolation, elongated stems and pale leaves, caused by chlorophyll loss and energy depletion. Prolonged darkness leads to carbohydrate starvation and death."],
            "retrieved": ["No, plants cannot survive or grow in complete darkness, as they require sunlight for photosynthesis."]
        },
        "Q10": {
            "query": "How should I design a raised bed garden to optimize drainage, root health, and overall yield?",
            "ground_truth": ["A well-designed raised bed should be 25–35 cm deep with layered soil: coarse gravel at the base for drainage, followed by loam and compost for aeration and nutrients. A slight 2–3% slope promotes runoff while preventing root rot and maintaining high yields."],
            "retrieved": ["Designing a raised bed garden to optimize drainage, root health, and overall yield involves several key considerations: 1. Choose the right soil: The soil you use in your raised bed garden should be well-draining, nutrient-rich, and pH-balanced. You can test your soil's pH and nutrient levels to determine if it needs any amendments. 2. Build the bed: The bed should be raised off the ground to allow for better drainage. You can use wood, brick, or concrete blocks to build the bed. Make sure the bed is at least 12 inches (30 cm) deep to accommodate root growth. 3. Add organic matter: Organic matter such as compost, manure, or leaf mold can help improve soil fertility, drainage, and root health. Add 2-4 inches (5-10 cm) of organic matter to the bed before planting. 4. Plant your vegetables: Choose vegetables that are well-suited to your climate and soil type. Plant them at the appropriate depth and spacing to ensure good growth and yield. 5. Water your vegetables: Water your vegetables regularly, but avoid overwatering. Water deeply and infrequently to encourage deep root growth. 6. Fertilize your vegetables: Fertilize your vegetables with a balanced fertilizer according to the manufacturer's instructions. Apply the fertilizer at the beginning of the growing season and again during the growing season."]
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