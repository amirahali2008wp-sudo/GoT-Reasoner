from openai import OpenAI
import json
import time
import os

class GoT_Reasoner:
    """
    یک چارچوب استدلال عمومی بر پایه گراف فکر برای حل مسائل پیچیده.
    """
    def __init__(self, client, model_name, problem_statement):
        self.client = client
        self.model_name = model_name
        self.problem = problem_statement
        self.graph = {"root": {"thought": "شروع مسئله", "score": 0, "parent": None, "op": "initial"}}
        self.node_counter = 0
        print(f"🚀 موتور استدلال گراف فکر راه‌اندازی شد با مدل: {self.model_name}")
        print(f"🤔 مسئله در حال حل: {self.problem}")

    def _call_llm(self, prompt_text, system_message="شما یک دستیار هوشمند و تحلیل‌گر هستید.", is_json=False):
        """فراخوانی مدل OpenAI."""
        try:
            response_format = {"type": "json_object"} if is_json else {"type": "text"}
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': prompt_text},
                ],
                temperature=0.7,
                response_format=response_format
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ خطایی در فراخوانی API رخ داد: {e}")
            return ""

    def _add_node(self, thought, score, parent="root", op="initial"):
        """افزودن یک فکر جدید به گراف."""
        node_id = f"node_{self.node_counter}"
        self.graph[node_id] = {"thought": thought, "score": score, "parent": parent, "operation": op}
        self.node_counter += 1
        print(f"✅ فکر جدید '{node_id}' با امتیاز {score:.2f} اضافه شد: '{thought[:50]}...'")

    def _needs_complex_reasoning(self):
        """
        تشخیص می‌دهد که آیا مسئله به استدلال چندمرحله‌ای نیاز دارد یا خیر.
        """
        prompt = f"""
        به سوال زیر دقت کن:
        "{self.problem}"
        آیا پاسخ به این سوال نیاز به فکر کردن چند مرحله‌ای، برنامه‌ریزی یا استدلال عمیق دارد، یا یک سوال ساده و واقعی (factual) است که می‌توان مستقیماً به آن پاسخ داد؟
        پاسخ را فقط با یک کلمه بده: YES یا NO.
        """
        response = self._call_llm(prompt, "شما یک متخصص طبقه‌بندی وظایف هستید.")
        return "YES" in response.upper()

    def evaluate_thought_quality(self, thought_candidate):
        """
        ارزیابی می‌کند که یک فکر چقدر به حل مسئله کمک می‌کند.
        """
        print(f"⏳ در حال ارزیابی فکر: '{thought_candidate[:60]}...'")

        evaluation_prompt = f"""
        صورت مسئله این است: "{self.problem}"
        فکر یا قدم پیشنهادی برای حل آن این است: "{thought_candidate}"

        به عنوان یک منطق‌دان، ارزیابی کن که این فکر چقدر به حل نهایی مسئله کمک می‌کند؟ آیا یک قدم معتبر و مفید است؟
        از ۱ تا ۱۰ به آن امتیاز بده.
        پاسخ را فقط در این فرمت JSON ارائه بده: {{"score": <عدد_امتیاز>, "reason": "<دلیل_مختصر>"}}
        """
        evaluator_response = self._call_llm(evaluation_prompt, "شما یک ارزیاب منطقی و سخت‌گیر هستید.", is_json=True)

        try:
            eval_json = json.loads(evaluator_response)
            return float(eval_json.get("score", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"⚠️ پاسخ ارزیاب نامعتبر بود: {evaluator_response}")
            return 0

    def generate_initial_thoughts(self, num_ideas=3):
        """تولید اولین ایده‌ها برای شروع حل مسئله."""
        prompt = f"""
        مسئله این است: "{self.problem}"
        برای شروع حل این مسئله، {num_ideas} ایده یا قدم اولیه متفاوت پیشنهاد بده.
        پاسخ باید یک لیست JSON از رشته‌ها باشد. مثال: ["قدم اول این است که...", "شاید بهتر باشد با..."]
        """
        ideas_response = self._call_llm(prompt, "شما یک ایده‌پرداز خلاق برای حل مسئله هستید.", is_json=True)
        try:
            thoughts = json.loads(ideas_response)
            if isinstance(thoughts, dict) and thoughts.get('thoughts'): thoughts = thoughts['thoughts'] # for compatibility

            for t in thoughts:
                score = self.evaluate_thought_quality(t)
                self._add_node(t, score)
        except (json.JSONDecodeError, TypeError, KeyError):
            print("❌ تولید ایده‌های اولیه ناموفق بود.")

    def refine_thought(self, parent_node_id):
        """پالایش و تکمیل یک فکر موجود."""
        parent_thought = self.graph[parent_node_id]["thought"]
        prompt = f"""
        مسئله: "{self.problem}"
        فکر فعلی این است: "{parent_thought}"
        این فکر را دقیق‌تر، کامل‌تر یا صحیح‌تر کن یا قدم منطقی بعدی را به آن اضافه نما.
        مهم: فقط متن فکر بهبودیافته را برگردان.
        """
        refined_thought = self._call_llm(prompt, "شما یک متفکر دقیق و منطقی هستید.")
        if refined_thought:
            score = self.evaluate_thought_quality(refined_thought)
            self._add_node(refined_thought, score, parent=parent_node_id, op="refine")

    def aggregate_thoughts(self, node_id_1, node_id_2):
        """ترکیب دو خط فکری متفاوت."""
        thought1 = self.graph[node_id_1]["thought"]
        thought2 = self.graph[node_id_2]["thought"]
        prompt = f"""
        مسئله: "{self.problem}"
        دو ایده برای حل آن وجود دارد:
        ایده الف: "{thought1}"
        ایده ب: "{thought2}"
        بهترین جنبه‌های هر دو ایده را ترکیب کن تا یک فکر یا راه‌حل ترکیبی قوی‌تر بسازی.
        مهم: فقط متن فکر جدید را برگردان.
        """
        aggregated_thought = self._call_llm(prompt, "شما یک استراتژیست و ترکیب‌کننده ایده‌ها هستید.")
        if aggregated_thought:
            score = self.evaluate_thought_quality(aggregated_thought)
            self._add_node(aggregated_thought, score, parent=[node_id_1, node_id_2], op="aggregate")

    def solve(self, cycles=3):
        """اجرای حلقه اصلی استدلال."""
        # تغییر کلیدی: بررسی اولیه نیاز به استدلال
        if not self._needs_complex_reasoning():
            print("\n🚦 نتیجه تشخیص: این سوال به استدلال پیچیده نیاز ندارد. (NO THINK)")
            # می‌توانید اینجا مستقیماً از مدل بخواهید به سوال ساده پاسخ دهد
            # direct_answer = self._call_llm(self.problem)
            # print(f"پاسخ مستقیم: {direct_answer}")
            return

        print("\n🚦 نتیجه تشخیص: این سوال به استدلال پیچیده نیاز دارد. شروع فرآیند گراف فکر...")

        print("\n--- مرحله ۱: تولید افکار اولیه ---")
        self.generate_initial_thoughts()

        for i in range(cycles):
            print(f"\n--- مرحله {i + 2}: شروع چرخه استدلال {i + 1}/{cycles} ---")
            sorted_nodes = sorted(self.graph.items(), key=lambda item: item[1]['score'], reverse=True)
            if len(sorted_nodes) < 2: break

            best_node_id = sorted_nodes[0][0]
            self.refine_thought(best_node_id)

            node1_id, node2_id = sorted_nodes[0][0], sorted_nodes[1][0]
            self.aggregate_thoughts(node1_id, node2_id)

        print("\n--- استدلال پایان یافت ---")
        self.show_best_solution()

    def show_best_solution(self):
        """نمایش بهترین فکر یا راه‌حل نهایی."""
        if len(self.graph) <= 1:
            print("هیچ راه‌حلی تولید نشد.")
            return

        # حذف گره ریشه از محاسبات
        valid_nodes = {k: v for k, v in self.graph.items() if k != 'root'}
        if not valid_nodes: return

        best_node = max(valid_nodes.items(), key=lambda item: item[1]['score'])
        print("\n🏆 بهترین فکر یا راه‌حل یافت‌شده:")
        print(f"   امتیاز: {best_node[1]['score']:.2f}")
        print(f"   راه‌حل: \"{best_node[1]['thought']}\"")

# --- نحوه اجرا ---
if __name__ == "__main__":
    try:
        # اتصال به API
        openai_client = OpenAI(
            
           
            api_key="API",
            base_url="سایت ارائه دهنده API",
        )

        # ۱. مسئله پیچیده خود را اینجا وارد کنید
        
        PROBLEM_STATEMENT ="مسئله پیچیده خود را اینجا وارد کنید"
        # ۲. ساخت و اجرای موتور استدلال
        reasoner = GoT_Reasoner(
            client=openai_client,
            model_name="نام مدل",
            problem_statement=PROBLEM_STATEMENT
        )
        reasoner.solve(cycles=2) # تعداد چرخه‌های فکر کردن

    except Exception as e:
        print(f"❌ یک خطای کلی در اجرای برنامه رخ داد: {e}")