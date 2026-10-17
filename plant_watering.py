import tkinter as tk
from tkinter import ttk, messagebox
import math
import random
import csv

# ---------- الجزء الأول: Perceptron ----------
def load_data(filename):
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # تخطي رأس العمود
            for row in reader:
                if len(row) < 4:
                    continue
                moisture = float(row[0])
                last_watered = float(row[1])
                plant_type = float(row[2])
                needs_water = int(row[3])
                data.append([moisture, last_watered, plant_type, needs_water])
    except Exception as e:
        raise Exception(f"فشل قراءة الملف '{filename}'. تأكد من أنه بصيغة CSV (وليس Excel حقيقي).\n"
                        f"الخطأ: {e}\n"
                        f"الحل: افتح ملف Excel واحفظه كـ CSV (UTF-8) ثم أعد المحاولة.")
    return data

def normalize_features(data):
    features = [row[:3] for row in data]
    n_features = len(features[0])
    min_vals = [float('inf')] * n_features
    max_vals = [-float('inf')] * n_features
    for row in features:
        for i in range(n_features):
            if row[i] < min_vals[i]: min_vals[i] = row[i]
            if row[i] > max_vals[i]: max_vals[i] = row[i]
    normalized = []
    for row in features:
        norm_row = []
        for i in range(n_features):
            if max_vals[i] - min_vals[i] == 0:
                norm_val = 0.0
            else:
                norm_val = (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            norm_row.append(norm_val)
        normalized.append(norm_row)
    data_normalized = []
    for i in range(len(data)):
        data_normalized.append(normalized[i] + [data[i][3]])
    return data_normalized, min_vals, max_vals

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []
        self.min_vals = None
        self.max_vals = None

    def train(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(n_features)]
        self.bias = 0.0
        for epoch in range(self.epochs):
            errors = 0
            for i in range(n_samples):
                linear = self.bias
                for j in range(n_features):
                    linear += self.weights[j] * X[i][j]
                pred = 1 if linear >= 0 else 0
                error = y[i] - pred
                if error != 0:
                    errors += 1
                    for j in range(n_features):
                        self.weights[j] += self.lr * error * X[i][j]
                    self.bias += self.lr * error
            self.loss_history.append(errors)
            acc = 1 - (errors / n_samples)
            self.accuracy_history.append(acc)
            if errors == 0:
                break

    def predict(self, X):
        X_norm = []
        for i in range(len(X)):
            norm_row = []
            for j in range(3):
                if self.max_vals[j] - self.min_vals[j] == 0:
                    norm_val = 0.0
                else:
                    norm_val = (X[i][j] - self.min_vals[j]) / (self.max_vals[j] - self.min_vals[j])
                norm_row.append(norm_val)
            X_norm.append(norm_row)
        preds = []
        for sample in X_norm:
            linear = self.bias
            for j in range(len(self.weights)):
                linear += self.weights[j] * sample[j]
            preds.append(1 if linear >= 0 else 0)
        return preds

    def set_normalization_params(self, min_vals, max_vals):
        self.min_vals = min_vals
        self.max_vals = max_vals

def train_perceptron_from_file(filename):
    raw = load_data(filename)
    norm_data, minv, maxv = normalize_features(raw)
    X_train = [row[:3] for row in norm_data]
    y_train = [row[3] for row in norm_data]
    p = Perceptron(learning_rate=0.1, epochs=100)
    p.set_normalization_params(minv, maxv)
    p.train(X_train, y_train)
    return p

# ---------- الجزء الثاني والثالث: GUI مع SA ----------
class WateringGUI:
    def __init__(self, root, perceptron):
        self.root = root
        self.perceptron = perceptron
        self.root.title("Smart Plant Watering Scheduler")
        self.root.geometry("1100x650")
        self.plants = []   # [id, x, y, moisture, last_watered, plant_type, needs_water]
        self.next_id = 1
        self.clicked_pos = None

        # Canvas
        self.canvas_frame = ttk.LabelFrame(root, text="Garden (click to add plant)")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.garden_canvas = tk.Canvas(self.canvas_frame, width=500, height=500, bg='lightgreen')
        self.garden_canvas.pack()
        self.garden_canvas.bind("<Button-1>", self.on_click)

        # Control panel
        self.control_frame = ttk.LabelFrame(root, text="Controls")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Soil Moisture (0-100):").pack(anchor='w')
        self.moisture_entry = ttk.Entry(self.control_frame)
        self.moisture_entry.pack(fill='x')
        ttk.Label(self.control_frame, text="Hours since last watering (0-48):").pack(anchor='w')
        self.watered_entry = ttk.Entry(self.control_frame)
        self.watered_entry.pack(fill='x')
        ttk.Label(self.control_frame, text="Plant Type:").pack(anchor='w')
        self.type_var = tk.IntVar(value=0)
        ttk.Radiobutton(self.control_frame, text="Cactus (0)", variable=self.type_var, value=0).pack(anchor='w')
        ttk.Radiobutton(self.control_frame, text="Flower (1)", variable=self.type_var, value=1).pack(anchor='w')
        ttk.Radiobutton(self.control_frame, text="Herb (2)", variable=self.type_var, value=2).pack(anchor='w')
        self.add_btn = ttk.Button(self.control_frame, text="Add Plant", command=self.add_plant)
        self.add_btn.pack(pady=5)

        self.test_btn = ttk.Button(self.control_frame, text="Test Perceptron", command=self.test_perceptron)
        self.test_btn.pack(fill='x', pady=2)

        self.curve_btn = ttk.Button(self.control_frame, text="Show Learning Curve", command=self.show_learning_curve)
        self.curve_btn.pack(fill='x', pady=2)

        ttk.Label(self.control_frame, text="SA Optimizer", font=('Arial',10,'bold')).pack(pady=5)
        ttk.Label(self.control_frame, text="Number to water:").pack(anchor='w')
        self.num_entry = ttk.Entry(self.control_frame)
        self.num_entry.pack(fill='x')
        self.sa_btn = ttk.Button(self.control_frame, text="Run SA", command=self.run_sa)
        self.sa_btn.pack(pady=5)

        self.info_text = tk.Text(self.control_frame, height=12, width=40)
        self.info_text.pack(fill='both', expand=True, pady=5)

    def on_click(self, event):
        self.clicked_pos = (event.x, event.y)
        self.info_text.insert(tk.END, f"Position ({event.x},{event.y}) saved. Enter details and click Add.\n")
        self.info_text.see(tk.END)

    def add_plant(self):
        if self.clicked_pos is None:
            messagebox.showerror("Error", "Click on garden first.")
            return
        try:
            m = float(self.moisture_entry.get())
            w = float(self.watered_entry.get())
            t = self.type_var.get()
            if not (0<=m<=100 and 0<=w<=48):
                raise ValueError
        except:
            messagebox.showerror("Error", "Invalid moisture (0-100) or hours (0-48).")
            return
        features = [[m, w, t]]
        needs = self.perceptron.predict(features)[0]
        pid = self.next_id
        self.next_id += 1
        x, y = self.clicked_pos
        self.plants.append([pid, x, y, m, w, t, needs])
        color = "red" if needs else "green"
        self.garden_canvas.create_oval(x-10, y-10, x+10, y+10, fill=color, outline="black")
        self.garden_canvas.create_text(x, y-12, text=str(pid), font=("Arial",8))
        self.info_text.insert(tk.END, f"Plant {pid} added. Needs water: {needs}\n")
        self.info_text.see(tk.END)
        self.moisture_entry.delete(0, tk.END)
        self.watered_entry.delete(0, tk.END)
        self.type_var.set(0)
        self.clicked_pos = None

    def test_perceptron(self):
        if not self.plants:
            messagebox.showinfo("Info", "No plants.")
            return
        feats = [[p[3], p[4], p[5]] for p in self.plants]
        preds = self.perceptron.predict(feats)
        correct = sum(1 for i in range(len(self.plants)) if preds[i] == self.plants[i][6])
        acc = correct/len(self.plants)*100
        messagebox.showinfo("Test", f"Accuracy: {acc:.1f}% ({correct}/{len(self.plants)})")

    def show_learning_curve(self):
        win = tk.Toplevel(self.root)
        win.title("Perceptron Learning Curve")
        canvas = tk.Canvas(win, width=600, height=400, bg='white')
        canvas.pack()
        loss = self.perceptron.loss_history
        if len(loss) < 2:
            canvas.create_text(300, 200, text="Not enough data to draw curve", font=("Arial",12))
        else:
            max_loss = max(loss)
            for i in range(1, len(loss)):
                x1 = (i-1)*500/(len(loss)-1) + 50
                y1 = 350 - (loss[i-1]/max_loss)*300
                x2 = i*500/(len(loss)-1) + 50
                y2 = 350 - (loss[i]/max_loss)*300
                canvas.create_line(x1, y1, x2, y2, fill='blue', width=2)
            canvas.create_text(300, 380, text="Epochs -> (blue = errors)", anchor='n')
        win.geometry("600x450")

    def dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def cost(self, order, needed_ids):
        needed_set = set(needed_ids)
        watered_set = set(order)
        missed = len(needed_set - watered_set)
        total_dist = 0.0
        cur = (0,0)
        for pid in order:
            for p in self.plants:
                if p[0] == pid:
                    nx, ny = p[1], p[2]
                    total_dist += self.dist(cur, (nx, ny))
                    cur = (nx, ny)
                    break
        extra = sum(1 for pid in order if pid not in needed_set)
        return 10*missed + 0.5*total_dist + 5*extra

    def simulated_annealing(self, plants_to_water, max_iter=5000, T0=1000, alpha=0.99):
        current = plants_to_water[:]
        random.shuffle(current)
        cur_cost = self.cost(current, plants_to_water)
        best = current[:]
        best_cost = cur_cost
        T = T0
        for _ in range(max_iter):
            new_order = current[:]
            i, j = random.sample(range(len(new_order)), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_cost = self.cost(new_order, plants_to_water)
            delta = new_cost - cur_cost
            if delta < 0 or random.random() < math.exp(-delta/T):
                current = new_order
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best = current[:]
                    best_cost = cur_cost
            T *= alpha
            if T < 1e-8:
                break
        return best, best_cost

    def run_sa(self):
        if not self.plants:
            messagebox.showerror("Error", "No plants.")
            return
        try:
            k = int(self.num_entry.get())
            if k <= 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Enter positive integer.")
            return
        needed = [p[0] for p in self.plants if p[6]==1]
        if not needed:
            messagebox.showinfo("Info", "No plant needs water.")
            return
        if k > len(needed):
            k = len(needed)
            messagebox.showinfo("Info", f"Only {len(needed)} need water. Using all.")
        random.shuffle(needed)
        subset = needed[:k]
        best_order, best_cost = self.simulated_annealing(subset)
        self.info_text.insert(tk.END, f"\nSA: Water {subset}\nOrder: {best_order}\nCost = {best_cost:.2f}\n")
        self.info_text.see(tk.END)
        self.garden_canvas.delete("path")
        cur = (0,0)
        for pid in best_order:
            for p in self.plants:
                if p[0]==pid:
                    nx, ny = p[1], p[2]
                    self.garden_canvas.create_line(cur[0], cur[1], nx, ny, fill='blue', width=2, tags="path")
                    cur = (nx, ny)
                    break

# ---------- التشغيل النهائي ----------
if __name__ == "__main__":
    try:
        perceptron = train_perceptron_from_file("Data.csv")
    except Exception as e:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("خطأ في قراءة الملف", str(e))
        root_err.destroy()
        exit(1)

    root = tk.Tk()
    app = WateringGUI(root, perceptron)
    root.mainloop()