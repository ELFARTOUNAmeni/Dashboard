
/** @odoo-module **/

import { registry } from "@web/core/registry"
import { useService } from "@web/core/utils/hooks"
import { Component, useState, onMounted, onWillUnmount, useRef } from "@odoo/owl"

class CustomerSegmentationDashboard extends Component {
  setup() {
    this.state = useState({
      dashboardData: null,
      loading: true,
      error: null,
      chartsRendered: false,
    })

    this.rpc = useService("rpc")
    this.action = useService("action")

    // Create individual refs for each chart
    this.segmentDistributionRef = useRef("segmentDistributionChart")
    this.monthlySalesRef = useRef("monthlySalesChart")
    this.segmentMetricsRef = useRef("segmentMetricsChart")

    this.charts = {}

    onMounted(() => {
      // Load dashboard data first
      this.loadDashboardData()
    })

    onWillUnmount(() => {
      // Destroy charts to prevent memory leaks
      Object.values(this.charts).forEach((chart) => {
        if (chart && typeof chart.destroy === "function") {
          chart.destroy()
        }
      })
    })
  }

  async loadDashboardData() {
    try {
      this.state.loading = true
      const data = await this.rpc("/sales_prediction/dashboard_data")
      this.state.dashboardData = data
      this.state.loading = false

      // Wait for the next rendering cycle before trying to render charts
      setTimeout(() => {
        this.loadChartJsAndRender()
      }, 100)
    } catch (error) {
      this.state.error = "Failed to load dashboard data"
      this.state.loading = false
      console.error("Dashboard data loading error:", error)
    }
  }

  async loadChartJsAndRender() {
    try {
      // Check if Chart is already defined
      if (typeof Chart === "undefined") {
        // Load Chart.js dynamically
        await new Promise((resolve, reject) => {
          const script = document.createElement("script")
          script.src = "/sales_prediction/static/lib/chart/chart.min.js"
          script.onload = resolve
          script.onerror = reject
          document.head.appendChild(script)
        })
      }

      // Now render the charts
      this.renderCharts()
    } catch (error) {
      console.error("Failed to load Chart.js:", error)
      this.state.error = "Failed to load chart library"
    }
  }

  renderCharts() {
    if (!this.state.dashboardData) return

    // Ensure the DOM elements are available
    if (this.segmentDistributionRef.el) {
      this.renderSegmentDistributionChart()
    } else {
      console.warn("Segment distribution chart element not found")
    }

    if (this.monthlySalesRef.el) {
      this.renderMonthlySalesChart()
    } else {
      console.warn("Monthly sales chart element not found")
    }

    if (this.segmentMetricsRef.el) {
      this.renderSegmentMetricsChart()
    } else {
      console.warn("Segment metrics chart element not found")
    }

    this.state.chartsRendered = true
  }

  renderSegmentDistributionChart() {
    try {
      const ctx = this.segmentDistributionRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for segment distribution chart")
        return
      }

      const labels = []
      const data = []
      const backgroundColors = []

      this.state.dashboardData.segments.forEach((segment) => {
        labels.push(segment.name)
        data.push(segment.customer_count)
        backgroundColors.push(segment.color || this.getRandomColor())
      })

      if (this.charts.segmentDistribution) {
        this.charts.segmentDistribution.destroy()
      }

      this.charts.segmentDistribution = new Chart(ctx, {
        type: "pie",
        data: {
          labels: labels,
          datasets: [
            {
              data: data,
              backgroundColor: backgroundColors,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "right",
            },
            tooltip: {
              callbacks: {
                label: (context) => {
                  const label = context.label || ""
                  const value = context.raw || 0
                  const dataset = context.dataset
                  const total = dataset.data.reduce((a, b) => a + b, 0)
                  const percentage = Math.floor((value / total) * 100 + 0.5)
                  return `${label}: ${value} customers (${percentage}%)`
                },
              },
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering segment distribution chart:", error)
    }
  }

  renderMonthlySalesChart() {
    try {
      const ctx = this.monthlySalesRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for monthly sales chart")
        return
      }

      const datasets = []

      this.state.dashboardData.segments.forEach((segment) => {
        const monthlyData = []

        this.state.dashboardData.months.forEach((month) => {
          monthlyData.push(this.state.dashboardData.monthly_sales[segment.id]?.[month] || 0)
        })

        datasets.push({
          label: segment.name,
          data: monthlyData,
          borderColor: segment.color || this.getRandomColor(),
          backgroundColor: (segment.color || this.getRandomColor()) + "33", // Add transparency
          fill: false,
          tension: 0.1,
        })
      })

      if (this.charts.monthlySales) {
        this.charts.monthlySales.destroy()
      }

      this.charts.monthlySales = new Chart(ctx, {
        type: "line",
        data: {
          labels: this.state.dashboardData.months || [],
          datasets: datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: "Sales Amount",
              },
            },
            x: {
              title: {
                display: true,
                text: "Month",
              },
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering monthly sales chart:", error)
    }
  }

  renderSegmentMetricsChart() {
    try {
      const ctx = this.segmentMetricsRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for segment metrics chart")
        return
      }

      const datasets = []

      this.state.dashboardData.segments.forEach((segment) => {
        // Check if metrics_distribution exists and has the required properties
        if (!this.state.dashboardData.metrics_distribution) {
          console.warn("metrics_distribution is missing in dashboard data")
          return
        }

        const metricsData = [
          this.state.dashboardData.metrics_distribution["avg_order_value"]?.[segment.id]?.avg || 0,
          this.state.dashboardData.metrics_distribution["order_frequency"]?.[segment.id]?.avg || 0,
          (this.state.dashboardData.metrics_distribution["total_spent"]?.[segment.id]?.avg || 0) / 1000, // Scale down for better visualization
          this.state.dashboardData.metrics_distribution["category_count"]?.[segment.id]?.avg || 0,
        ]

        datasets.push({
          label: segment.name,
          data: metricsData,
          borderColor: segment.color || this.getRandomColor(),
          backgroundColor: (segment.color || this.getRandomColor()) + "33", // Add transparency
        })
      })

      if (this.charts.segmentMetrics) {
        this.charts.segmentMetrics.destroy()
      }

      this.charts.segmentMetrics = new Chart(ctx, {
        type: "radar",
        data: {
          labels: ["Avg Order Value", "Order Frequency", "Total Spent (K)", "Category Count"],
          datasets: datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          elements: {
            line: {
              borderWidth: 3,
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering segment metrics chart:", error)
    }
  }

  // Helper function to generate random colors for segments that don't have a color
  getRandomColor() {
    const letters = "0123456789ABCDEF"
    let color = "#"
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)]
    }
    return color
  }

  onRefreshDashboard() {
    this.loadDashboardData()
  }

  onGenerateSegments() {
    this.action.doAction({
      name: "Generate Customer Segments",
      type: "ir.actions.act_window",
      res_model: "generate.customer.segments.wizard",
      views: [[false, "form"]],
      target: "new",
      context: {},
    })
  }

  onViewSegment(segmentId) {
    this.action.doAction({
      name: "Customer Segment",
      type: "ir.actions.act_window",
      res_model: "customer.segment",
      res_id: segmentId,
      views: [[false, "form"]],
      target: "current",
    })
  }

  onViewCustomers(segmentId) {
    this.action.doAction({
      name: "Customers",
      type: "ir.actions.act_window",
      res_model: "res.partner",
      domain: [["segment_id", "=", segmentId]],
      views: [
        [false, "list"],
        [false, "form"],
      ],
      target: "current",
    })
  }

  onViewProduct(productId) {
    this.action.doAction({
      name: "Product",
      type: "ir.actions.act_window",
      res_model: "product.product",
      res_id: productId,
      views: [[false, "form"]],
      target: "current",
    })
  }
}

CustomerSegmentationDashboard.template = "sales_prediction.CustomerSegmentationDashboard"

registry.category("actions").add("customer_segmentation_dashboard", CustomerSegmentationDashboard)

export default CustomerSegmentationDashboard
