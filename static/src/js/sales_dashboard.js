
/** @odoo-module **/

import { registry } from "@web/core/registry"
import { useService } from "@web/core/utils/hooks"
import { Component, useState, onMounted, onWillUnmount, useRef } from "@odoo/owl"

class SalesDashboard extends Component {
  setup() {
    const initialFilters = {
      dateFrom: this.getDateString(new Date(new Date().setDate(new Date().getDate() - 30))),
      dateTo: this.getDateString(new Date()),
      warehouseId: false,
      categoryId: false,
    }

    this.state = useState({
      dashboardData: null,
      loading: true,
      error: null,
      chartsRendered: false,
      filters: initialFilters,
    })

    this.rpc = useService("rpc")
    this.action = useService("action")

    // Create refs for the chart canvases
    this.salesTrendRef = useRef("salesTrendChart")
    this.categoryDistributionRef = useRef("categoryDistributionChart")
    this.warehouseDistributionRef = useRef("warehouseDistributionChart")
    this.stockStatusRef = useRef("stockStatusChart")
    this.seasonalSalesRef = useRef("seasonalSalesChart")
    this.dayOfWeekRef = useRef("dayOfWeekChart")

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

  getDateString(date) {
    return date.toISOString().split("T")[0]
  }

  async loadDashboardData() {
    try {
      this.state.loading = true
      const data = await this.rpc("/sales_prediction/sales_dashboard_data", {
        date_from: this.state.filters.dateFrom,
        date_to: this.state.filters.dateTo,
        warehouse_id: this.state.filters.warehouseId,
        category_id: this.state.filters.categoryId,
      })

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
          script.src = "/sales_prediction/static/lib/chart.js/chart.min.js"
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
    if (this.salesTrendRef.el) {
      this.renderSalesTrendChart()
    } else {
      console.warn("Sales trend chart element not found")
    }

    if (this.categoryDistributionRef.el) {
      this.renderCategoryDistributionChart()
    } else {
      console.warn("Category distribution chart element not found")
    }

    if (this.warehouseDistributionRef.el) {
      this.renderWarehouseDistributionChart()
    } else {
      console.warn("Warehouse distribution chart element not found")
    }

    if (this.stockStatusRef.el) {
      this.renderStockStatusChart()
    } else {
      console.warn("Stock status chart element not found")
    }

    if (this.seasonalSalesRef.el) {
      this.renderSeasonalSalesChart()
    } else {
      console.warn("Seasonal sales chart element not found")
    }

    if (this.dayOfWeekRef.el) {
      this.renderDayOfWeekChart()
    } else {
      console.warn("Day of week chart element not found")
    }

    this.state.chartsRendered = true
  }

  renderSalesTrendChart() {
    try {
      const ctx = this.salesTrendRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for sales trend chart")
        return
      }

      if (this.charts.salesTrend) {
        this.charts.salesTrend.destroy()
      }

      this.charts.salesTrend = new Chart(ctx, {
        type: "line",
        data: {
          labels: this.state.dashboardData.trend_dates || [],
          datasets: [
            {
              label: "Sales Amount",
              data: this.state.dashboardData.trend_amounts || [],
              borderColor: "#3f51b5",
              backgroundColor: "rgba(63, 81, 181, 0.1)",
              borderWidth: 2,
              fill: true,
              tension: 0.4,
            },
            {
              label: "Quantity Sold",
              data: this.state.dashboardData.trend_quantities || [],
              borderColor: "#f44336",
              backgroundColor: "rgba(244, 67, 54, 0.1)",
              borderWidth: 2,
              fill: true,
              tension: 0.4,
              yAxisID: "y1",
            },
          ],
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
            y1: {
              beginAtZero: true,
              position: "right",
              grid: {
                drawOnChartArea: false,
              },
              title: {
                display: true,
                text: "Quantity Sold",
              },
            },
            x: {
              title: {
                display: true,
                text: "Date",
              },
            },
          },
          plugins: {
            tooltip: {
              mode: "index",
              intersect: false,
            },
            legend: {
              position: "top",
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering sales trend chart:", error)
    }
  }

  renderCategoryDistributionChart() {
    try {
      const ctx = this.categoryDistributionRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for category distribution chart")
        return
      }

      const labels = []
      const data = []
      const backgroundColors = []

      if (this.state.dashboardData.categories) {
        this.state.dashboardData.categories.forEach((category) => {
          labels.push(category.name)
          data.push(category.amount_sold)
          backgroundColors.push(category.color || this.getRandomColor())
        })
      }

      if (this.charts.categoryDistribution) {
        this.charts.categoryDistribution.destroy()
      }

      this.charts.categoryDistribution = new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: labels,
          datasets: [
            {
              data: data,
              backgroundColor: backgroundColors,
              borderWidth: 1,
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
                  return `${label}: ${this.formatCurrency(value)} (${percentage}%)`
                },
              },
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering category distribution chart:", error)
    }
  }

  renderWarehouseDistributionChart() {
    try {
      const ctx = this.warehouseDistributionRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for warehouse distribution chart")
        return
      }

      const labels = []
      const data = []
      const backgroundColors = []

      if (this.state.dashboardData.warehouses) {
        this.state.dashboardData.warehouses.forEach((warehouse) => {
          labels.push(warehouse.name)
          data.push(warehouse.amount_sold)
          backgroundColors.push(warehouse.color || this.getRandomColor())
        })
      }

      if (this.charts.warehouseDistribution) {
        this.charts.warehouseDistribution.destroy()
      }

      this.charts.warehouseDistribution = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Sales Amount",
              data: data,
              backgroundColor: backgroundColors,
              borderWidth: 1,
            },
          ],
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
          },
          plugins: {
            legend: {
              display: false,
            },
            tooltip: {
              callbacks: {
                label: (context) => {
                  return `Sales: ${this.formatCurrency(context.raw)}`
                },
              },
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering warehouse distribution chart:", error)
    }
  }

  renderStockStatusChart() {
    try {
      const ctx = this.stockStatusRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for stock status chart")
        return
      }

      const labels = []
      const data = []
      const backgroundColors = {
        low: "#f44336",
        normal: "#4caf50",
        overstock: "#ff9800",
        no_sales: "#9e9e9e",
        unknown: "#607d8b",
      }
      const colors = []

      if (this.state.dashboardData.stock_status) {
        this.state.dashboardData.stock_status.forEach((status) => {
          labels.push(status.status_name)
          data.push(status.sales_quantity)
          colors.push(backgroundColors[status.status] || backgroundColors.unknown)
        })
      }

      if (this.charts.stockStatus) {
        this.charts.stockStatus.destroy()
      }

      this.charts.stockStatus = new Chart(ctx, {
        type: "pie",
        data: {
          labels: labels,
          datasets: [
            {
              data: data,
              backgroundColor: colors,
              borderWidth: 1,
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
                  return `${label}: ${value} units (${percentage}%)`
                },
              },
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering stock status chart:", error)
    }
  }

  renderSeasonalSalesChart() {
    try {
      const ctx = this.seasonalSalesRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for seasonal sales chart")
        return
      }

      const labels = []
      const data = []
      const backgroundColors = []

      if (this.state.dashboardData.seasons) {
        this.state.dashboardData.seasons.forEach((season) => {
          labels.push(season.season_name)
          data.push(season.total_sales)
          backgroundColors.push(season.color || this.getRandomColor())
        })
      }

      if (this.charts.seasonalSales) {
        this.charts.seasonalSales.destroy()
      }

      this.charts.seasonalSales = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Total Sales",
              data: data,
              backgroundColor: backgroundColors,
              borderWidth: 1,
            },
          ],
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
          },
          plugins: {
            legend: {
              display: false,
            },
            tooltip: {
              callbacks: {
                label: (context) => {
                  return `Sales: ${this.formatCurrency(context.raw)}`
                },
              },
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering seasonal sales chart:", error)
    }
  }

  renderDayOfWeekChart() {
    try {
      const ctx = this.dayOfWeekRef.el.getContext("2d")
      if (!ctx) {
        console.error("Could not get 2D context for day of week chart")
        return
      }

      const labels = []
      const data = []
      const backgroundColors = []

      if (this.state.dashboardData.days_of_week) {
        this.state.dashboardData.days_of_week.forEach((day) => {
          labels.push(day.name)
          data.push(day.total_sales)
          backgroundColors.push(day.color || this.getRandomColor())
        })
      }

      if (this.charts.dayOfWeek) {
        this.charts.dayOfWeek.destroy()
      }

      this.charts.dayOfWeek = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Total Sales",
              data: data,
              backgroundColor: backgroundColors,
              borderWidth: 1,
            },
          ],
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
          },
          plugins: {
            legend: {
              display: false,
            },
            tooltip: {
              callbacks: {
                label: (context) => {
                  return `Sales: ${this.formatCurrency(context.raw)}`
                },
              },
            },
          },
        },
      })
    } catch (error) {
      console.error("Error rendering day of week chart:", error)
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

  // Helper function to format currency values
  formatCurrency(value) {
    return value.toLocaleString("en-US", {
      style: "currency",
      currency: "USD",
    })
  }

  onRefreshDashboard() {
    this.loadDashboardData()
  }

  onFilterChange() {
    this.loadDashboardData()
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

  onViewCategory(categoryId) {
    this.action.doAction({
      name: "Product Category",
      type: "ir.actions.act_window",
      res_model: "product.category",
      res_id: categoryId,
      views: [[false, "form"]],
      target: "current",
    })
  }

  onViewWarehouse(warehouseId) {
    this.action.doAction({
      name: "Warehouse",
      type: "ir.actions.act_window",
      res_model: "stock.warehouse",
      res_id: warehouseId,
      views: [[false, "form"]],
      target: "current",
    })
  }

  onExportData() {
    // Implement export functionality
    console.log("Export data")
  }
}

SalesDashboard.template = "sales_prediction.SalesDashboard"

registry.category("actions").add("sales_dashboard", SalesDashboard)

export default SalesDashboard
